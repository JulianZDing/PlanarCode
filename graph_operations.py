import numpy as np
import networkx as nx

from collections import defaultdict
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
from networkx.algorithms.matching import max_weight_matching
from planar_code import PlanarLattice


ANCILLA_WEIGHT = 0
MANHATTAN = 'manhattan'
DIJKSTRA = 'dijkstra'
WEIGHT_KEY = 'weight'

def p_to_dist(p):
    return np.log((1-p) / p)

def is_real_site(lattice, v):
    L, W = lattice.shape
    x, y = v
    return x < L and y < W


def lattice_to_graph(lattice):
    '''Represent a lattice as a networkx Graph
    
    Arguments:
    lattice -- PlanarLattice instance

    Returns nx.Graph instance
    '''
    X, Y, D = lattice.edges.shape
    graph = nx.Graph()
    for x in range(X):
        for y in range(Y):
            for d in range(D):
                vx, vy = lattice.edge_endpoints[:, x, y, d]
                v0x, v1x = vx
                v0y, v1y = vy
                v0 = (v0x, v0y)
                v1 = (v1x, v1y)
                if not is_real_site(lattice, v1):
                    continue
                v0_is_site = lattice.sites[v0]
                v1_is_site = lattice.sites[v1]
                if v0_is_site or v1_is_site:
                    weight = p_to_dist(lattice.p[x, y, d])
                    graph.add_edge(v0, v1, weight=weight)
    return graph


def min_weight_syndrome_matching(lattice, syndrome, pathfinding=None):
    '''Match syndromes along paths of maximum error probability

    Arguments:
    lattice -- PlanarLattice instance
    syndrome -- lattice.shape boolean array marking nontrivial syndromes

    Positional arguments:
    pathfinding -- whether to force dijkstra/Manhattan pathfinding
                   (default: None; options: "dijkstra", "manhattan")

    Returns a list of pairs of coordinates corresponding to matching syndromes
    and a corresponding list of paths to complete each pairing
    '''
    matching_graph, paths = syndrome_to_matching_graph(lattice, syndrome, pathfinding)
    # Transform from min weight to max weight matching problem
    max_weight = 0
    for v0, v1, weight in matching_graph.edges.data(WEIGHT_KEY):
        max_weight = max(max_weight, weight)
    for v0, v1, weight in matching_graph.edges.data(WEIGHT_KEY):
        prev_weight = matching_graph[v0][v1][WEIGHT_KEY]
        matching_graph[v0][v1][WEIGHT_KEY] = max_weight - prev_weight + 1
    # Perform matching
    matched_labels = max_weight_matching(matching_graph, maxcardinality=True)
    # Prune paths to ancilla nodes
    matching = []
    for pair in matched_labels:
        matched_pair = []
        for coords in pair:
            if is_real_site(lattice, coords):
                matched_pair.append(coords)
            else:
                ancilla = coords
        if matched_pair:
            if len(matched_pair) < 2:
                matched_pair.append(ancilla)
                i, j = matched_pair
                for k in [0, -1]:
                    if not is_real_site(lattice, paths[i][j][k]):
                        del paths[i][j][k]
            matching.append(matched_pair)
    return matching, [paths[i][j] for i, j in matching]


def syndrome_to_matching_graph(lattice, syndrome, pathfinding=None):
    '''Represent a syndrome as a fully connected graph

    Sites with nontrivial syndrome measurements are connected via
    the minimum-weight paths between sites

    If there are rough boundaries, a number of "rough vertices"
    equivalent to the number of syndrome sites are added to the graph
    connected to each of the rough boundary endpoints. Edges between
    these endpoints and between rough vertices have weight 0

    Arguments:
    lattice -- PlanarLattice instance
    syndrome -- lattice.shape boolean array marking nontrivial syndromes

    Positional arguments:
    pathfinding -- whether to force dijkstra/Manhattan pathfinding
                   (default: None; options: "dijkstra", "manhattan")

    Returns translated graph as nx.Graph instance,
    and paths dictionary with the lowest-distance paths between all syndromes
    '''
    xs, ys = lattice.grid[:, syndrome]
    syndrome_coords = list(zip(xs, ys))
    # Use Manhattan distance pathfinding by default
    # if error probabilities are uniform
    if lattice.uniform_p is not None:
        graph_func = manhattan_graph
    else:
        graph_func = dijkstra_graph
    if pathfinding == MANHATTAN:
        graph_func = manhattan_graph
    elif pathfinding == DIJKSTRA:
        graph_func = dijkstra_graph
    return graph_func(lattice, syndrome_coords)


def manhattan_graph(lattice, coords):
    '''Generate a fully connected graph between syndrome coordinates

    Assumes uniform error probability and uses Manhattan distance
    to calculate distance between sites.
    '''
    graph = nx.Graph()
    paths = defaultdict(dict)
    for i, v0 in enumerate(coords[:-1]):
        for v1 in coords[i+1:]:
            dist, path = manhattan_distance(
                lattice, v0, v1, make_path=True)
            update_paths(graph, paths, v0, v1, dist, path)
    rough_vertices = get_rough_vertices(lattice, len(coords))
    if rough_vertices:
        # Connect rough vertices to each other
        add_ancilla_nodes(graph, rough_vertices)
        # Connect rough vertices to rest of vertices
        rv0 = rough_vertices[0]
        for v in coords:
            dist, path = manhattan_distance(
                lattice, rv0, v, make_path=True)
            for rv in rough_vertices:
                update_paths(graph, paths, rv, v, dist, path)
    return graph, paths


def manhattan_distance(lattice, v0, v1, make_path=False):
    if not is_real_site(lattice, v0):
        v0 = manhattan_nearest_rough(lattice, v1)
    if not is_real_site(lattice, v1):
        v1 = manhattan_nearest_rough(lattice, v0)
    deltas = []
    directions = []
    for d in range(lattice.shape.size):
        start = v0[d]
        stop = v1[d]
        delta = abs(stop - start)
        direction = 1 if stop > start else -1
        # Handle periodic boundaries
        if lattice.boundaries[d] == PlanarLattice.PERIODIC:
            if stop > start:
                wrap_dist = start + lattice.shape[d]-1-stop
            else:
                wrap_dist = stop + lattice.shape[d]-1-stop
            if wrap_dist < delta:
                delta = wrap_dist
                direction *= -1
        deltas.append(delta)
        directions.append(direction)
    if not make_path:
        return sum(deltas)
    path = [v0]
    base = list(v0)
    for d in range(lattice.shape.size):
        direction = directions[d]
        start = v0[d]
        for i in range(deltas[d]):
            base[d] = start + direction*(i+1)
            path.append(tuple(base))
    return sum(deltas), path


def manhattan_nearest_rough(lattice, v):
    if not is_real_site(lattice, v):
        return v
    for d in range(lattice.shape.size):
        if lattice.boundaries[d] != PlanarLattice.ROUGH:
            continue
        L = lattice.shape[d]
        x = v[d]
        deltas = [(x, -1), (L-1-x, 1)]
        break
    rv = list(v)
    moves, direction = min(deltas)
    rv[d] += moves * direction
    return tuple(rv)


def dijkstra_graph(lattice, homes):
    '''Generate a fully connected graph between syndrome coordinates

    Uses dijkstra's algorithm to find the minimum distance between all coords marked home.
    '''
    lattice_graph = lattice_to_graph(lattice)
    reduced_graph = nx.Graph()
    paths = defaultdict(dict)
    for i, source in enumerate(homes[:-1]):
        for target in homes[i+1:]:
            distance, path = single_source_dijkstra(
                lattice_graph, source, target)
            update_paths(
                reduced_graph, paths, source, target, distance, path)
    rough_vertices = get_rough_vertices(lattice, len(homes))
    if rough_vertices:
        # Connect rough vertices to each other
        add_ancilla_nodes(reduced_graph, rough_vertices)
        # Connect rough vertices to rest of graph
        rv0 = rough_vertices[0]
        xs, ys = lattice.grid[:, lattice.sites == False]
        for b in zip(xs, ys):
            lattice_graph.add_edge(rv0, b, weight=ANCILLA_WEIGHT)
            for target in homes:
                distance, path = single_source_dijkstra(
                    lattice_graph, rv0, target)
                for rv in rough_vertices:
                    update_paths(
                        reduced_graph, paths, rv, target, distance, path)
    return reduced_graph, paths


def update_paths(graph, paths, v0, v1, dist, path):
    graph.add_edge(v0, v1, weight=dist)
    paths[v0][v1] = path
    paths[v1][v0] = path


def add_ancilla_nodes(graph, rough_vertices):
    ancilla = generate_rough_cluster(rough_vertices)
    graph.add_weighted_edges_from(ancilla.edges.data(WEIGHT_KEY))


def generate_rough_cluster(vertex_labels, weight=ANCILLA_WEIGHT):
    cluster = nx.Graph()
    n = len(vertex_labels)
    for i in range(n-1):
        v0 = vertex_labels[i]
        for j in range(i+1, n):
            v1 = vertex_labels[j]
            cluster.add_edge(v0, v1, weight=weight)
    return cluster


def get_rough_vertices(lattice, how_many):
    if np.any(lattice.boundaries == PlanarLattice.ROUGH):
        L, W = lattice.shape
        return [(L,W+i) for i in range(how_many)]
    return None
