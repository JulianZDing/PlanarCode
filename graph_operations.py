import numpy as np
import networkx as nx

from collections import defaultdict
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
from networkx.algorithms.matching import max_weight_matching

ANCILLA_WEIGHT = 0
MANHATTAN = 'manhattan'
DIJKSTRA = 'dijkstra'
WEIGHT_KEY = 'weight'

def p_to_dist(p):
    if p > 0:
        return np.log((1-p) / p)
    return 0

def _min_to_max_weight(graph):
    max_weight = 0
    for v0, v1, weight in graph.edges.data(WEIGHT_KEY):
        max_weight = max(max_weight, weight)
    for v0, v1, weight in graph.edges.data(WEIGHT_KEY):
        prev_weight = graph[v0][v1][WEIGHT_KEY]
        graph[v0][v1][WEIGHT_KEY] = max_weight - prev_weight + 1


def min_weight_syndrome_matching(lattice, syndrome, pathfinding=None, **kwargs):
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
    matching_graph, paths = syndrome_to_matching_graph(
        lattice, syndrome, pathfinding, **kwargs)
    # Transform from min weight to max weight matching problem (in-place)
    _min_to_max_weight(matching_graph)
    # Perform matching
    matched_labels = max_weight_matching(matching_graph, maxcardinality=True)
    # Prune paths to ancilla nodes
    matching = []
    for pair in matched_labels:
        matched_pair = []
        for coords in pair:
            if lattice.is_real_site(coords):
                matched_pair.append(coords)
            else:
                ancilla = coords
        if matched_pair:
            if len(matched_pair) < 2:
                matched_pair.append(ancilla)
                i, j = matched_pair
                for k in [0, -1]:
                    if not lattice.is_real_site(paths[i][j][k]):
                        del paths[i][j][k]
            matching.append(matched_pair)
    return matching, [paths[i][j] for i, j in matching]


def syndrome_to_matching_graph(lattice, syndrome, pathfinding=None, **kwargs):
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
    # Use Manhattan distance pathfinding if error probabilities are uniform
    if lattice.uniform_p is not None:
        graph_func = manhattan_graph
    else:
        graph_func = dijkstra_graph
    if pathfinding == MANHATTAN:
        graph_func = manhattan_graph
    elif pathfinding == DIJKSTRA:
        graph_func = dijkstra_graph
    graph, paths = graph_func(lattice, syndrome_coords, **kwargs)
    return graph, paths


def manhattan_graph(lattice, coords):
    '''Generate a fully connected graph between syndrome coordinates

    Assumes uniform error probability and uses Manhattan distance
    to calculate distance between sites.
    '''
    graph = nx.Graph()
    paths = defaultdict(dict)
    for i, v0 in enumerate(coords[:-1]):
        for v1 in coords[i+1:]:
            dist, path = manhattan_distance(lattice, v0, v1)
            update_paths(graph, paths, v0, v1, dist, path)
    # Connect rough vertices to each other
    rough_vertices = get_rough_vertices(lattice, len(coords))
    if len(rough_vertices) > 0:
        add_ancilla_nodes(graph, rough_vertices, ANCILLA_WEIGHT)
        # Connect rough vertices to rest of vertices
        rv0 = rough_vertices[0]
        for v in coords:
            dist, path = manhattan_distance(lattice, rv0, v)
            for rv in rough_vertices:
                update_paths(graph, paths, rv, v, dist, path)
    return graph, paths


def manhattan_distance(lattice, v0, v1):
    if not lattice.is_real_site(v0):
        v0 = manhattan_nearest_rough(lattice, v1)
    if not lattice.is_real_site(v1):
        v1 = manhattan_nearest_rough(lattice, v0)
    deltas = []
    directions = []
    for d in range(lattice.shape.size):
        start = v0[d]
        stop = v1[d]
        delta = abs(stop - start)
        direction = 1 if stop > start else -1
        deltas.append(delta)
        directions.append(direction)
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
    if not lattice.is_real_site(v):
        return v
    for d in range(lattice.shape.size):
        if lattice.boundaries[d]:
            continue
        L = lattice.shape[d]
        x = v[d]
        deltas = [(x, -1), (L-1-x, 1)]
        break
    rv = list(v)
    moves, direction = min(deltas)
    rv[d] += moves * direction
    return tuple(rv)


def dijkstra_graph(lattice, homes, **kwargs):
    '''Generate a fully connected graph between syndrome coordinates

    Uses Dijkstra's algorithm to find the minimum distance between all coords marked home.
    '''
    lattice_graph = lattice.to_graph(p_scaling_fn=p_to_dist, **kwargs)
    # Compute shortest paths between syndrome sites
    reduced_graph = nx.Graph()
    paths = defaultdict(dict)
    for i, source in enumerate(homes[:-1]):
        distances, these_paths = single_source_dijkstra(lattice_graph, source)
        for target in homes[i+1:]:
            distance = distances[target]
            path = these_paths[target]
            update_paths(reduced_graph, paths, source, target, distance, path)
    # Connect rough vertices to each other
    rough_vertices = get_rough_vertices(lattice, len(homes))
    if len(rough_vertices) > 0:
        add_ancilla_nodes(reduced_graph, rough_vertices, ANCILLA_WEIGHT)
        # Connect rough vertex to rough boundaries
        rv0 = rough_vertices[0]
        xs, ys = lattice.grid[:, lattice.sites == False]
        for b in zip(xs, ys):
            lattice_graph.add_edge(rv0, b, weight=ANCILLA_WEIGHT)
        # Compute shortest paths from syndrome sites to boundaries
        distances, rough_paths = single_source_dijkstra(lattice_graph, rv0)
        for target in homes:
            distance = distances[target]
            path = rough_paths[target]
            for rv in rough_vertices:
                update_paths(reduced_graph, paths, rv, target, distance, path)
    return reduced_graph, paths


def update_paths(graph, paths, v0, v1, dist, path):
    graph.add_edge(v0, v1, weight=dist)
    paths[v0][v1] = path
    paths[v1][v0] = path[::-1]


def add_ancilla_nodes(graph, rough_vertices, ancilla_weight):
    ancilla = generate_rough_cluster(rough_vertices, ancilla_weight)
    graph.add_weighted_edges_from(ancilla.edges.data(WEIGHT_KEY))


def generate_rough_cluster(vertex_labels, weight):
    cluster = nx.Graph()
    n = len(vertex_labels)
    for i in range(n-1):
        v0 = vertex_labels[i]
        for j in range(i+1, n):
            v1 = vertex_labels[j]
            cluster.add_edge(v0, v1, weight=weight)
    return cluster


def get_rough_vertices(lattice, how_many):
    L, W = lattice.shape
    return [(L,W+i) for i in range(how_many)]
