import numpy as np
import networkx as nx

from collections import defaultdict
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
from networkx.algorithms.matching import min_weight_matching
from planar_code import PlanarCode, PlanarLattice


ANCILLA_WEIGHT = 0.5


def lattice_to_graph(lattice):
    '''Represent a lattice as a networkx Graph
    
    Arguments:
    lattice -- PlanarLattice instance

    Returns nx.Graph instance
    '''
    L, W = lattice.shape
    X, Y, D = lattice.edges.shape
    graph = nx.Graph()
    for x in range(X):
        for y in range(Y):
            for d in range(D):
                vx, vy = lattice.edge_endpoints[:, x, y, d]
                v0x, v1x = vx
                v0y, v1y = vy
                if v1x >= L or v1y >=W:
                    continue
                v0_is_site = lattice.sites[v0x, v0y]
                v1_is_site = lattice.sites[v1x, v1y]
                if v0_is_site or v1_is_site:
                    v0 = (v0x, v0y)
                    v1 = (v1x, v1y)
                    weight = p_to_dist(lattice.p[x, y, d])
                    graph.add_edge(v0, v1, weight=weight)
    return graph


def min_weight_syndrome_matching(lattice, syndrome):
    '''Match syndromes along paths of maximum error probability

    Arguments:
    lattice -- PlanarLattice instance
    syndrome -- lattice.shape boolean array marking nontrivial syndromes

    Returns a list of pairs of coordinates corresponding to matching syndromes
    and a corresponding list of paths to complete each pairing
    '''
    L, W = lattice.shape
    matching_graph, paths = syndrome_to_matching_graph(lattice, syndrome)
    matched_labels = min_weight_matching(matching_graph, maxcardinality=True)
    matching = []
    for pair in matched_labels:
        matched_pair = []
        for coords in pair:
            x, y = coords
            if x < L and y < W:
                matched_pair.append(coords)
            else:
                ancilla = coords
        if matched_pair:
            if len(matched_pair) < 2:
                matched_pair.append(ancilla)
                i, j = matched_pair
                if paths[i][j][0] == ancilla:
                    del paths[i][j][0]
                else:
                    del paths[i][j][-1]
            matching.append(matched_pair)
    return matching, [paths[i][j] for i, j in matching]


def syndrome_to_matching_graph(lattice, syndrome, force_manhattan=False):
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
    force_manhattan -- use Manhattan distance calculation regardless of error model

    Returns translated graph as nx.Graph instance,
    and paths dictionary with the lowest-distance paths between all syndromes
    '''
    L, W = lattice.shape
    xs, ys = lattice.grid[:, syndrome]
    syndrome_coords = list(zip(xs, ys))
    # Use Manhattan distance if error probabilities are uniform
    if lattice.uniform_p is not None or force_manhattan:
        graph, paths = manhattan_graph(lattice, syndrome_coords)
    # Otherwise, use Dijkstra's algorithm
    else:
        graph, paths = dijkstra_graph(lattice, syndrome_coords)
    return graph, paths


def manhattan_graph(lattice, coords):
    '''Generate a fully connected graph between syndrome coordinates

    Assumes uniform error probability and uses Manhattan distance
    to scale edge weights.
    '''
    L, W = lattice.shape
    graph = nx.Graph()
    paths = defaultdict(dict)
    def add_path(v0, v1):
        weight, path = manhattan_distance(lattice, v0, v1, make_path=True)
        graph.add_edge(v0, v1, weight=weight)
        paths[v0][v1] = path
        paths[v1][v0] = path

    for i, v0 in enumerate(coords[:-1]):
        for v1 in coords[i+1:]:
            add_path(v0,v1)
    rough_vertices = get_rough_vertices(lattice, len(coords))
    if rough_vertices:
        # Connect rough vertices to each other
        add_ancilla_nodes(graph, rough_vertices)
        # Connect rough vertices to rest of vertices
        for rv in rough_vertices:
            for v in coords:
                add_path(rv, v)
    return graph, paths


def manhattan_distance(lattice, v0, v1, make_path=False):
    deltas = []
    directions = []
    for d in range(PlanarLattice.D):
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
    path = []
    base = list(v0)
    for d in range(PlanarLattice.D):
        direction = directions[d]
        start = v0[d]
        this_base = base
        for i in range(deltas[d]):
            this_base[d] = start + direction*(i+1)
            path.append(list(this_base))
        base[d] = v1[d]
    return sum(deltas), path


def manhattan_dist_to_rough(lattice, v0):
    L, W = lattice.shape
    x, y = v0
    horizontal, vertical = (lattice.boundaries == PlanarLattice.ROUGH)
    deltas = []
    if horizontal:
        deltas.append(x)
        deltas.append(L-1-x)
    if vertical:
        deltas.append(y)
        deltas.append(W-1-y)
    return min(deltas)


def dijkstra_graph(lattice, homes):
    '''Generate a fully connected graph between syndrome coordinates

    Uses dijkstra's algorithm to find the minimum distance between all coords marked home.
    '''

    L, W = lattice.shape
    lattice_graph = lattice_to_graph(lattice)
    reduced_graph = nx.Graph()
    paths = defaultdict(dict)
    def compute_dijkstra(source, target):
        distance, path = single_source_dijkstra(lattice_graph, source, target)
        paths[source][target] = path
        paths[target][source] = path
        reduced_graph.add_edge(source, target, weight=distance)

    for i, source in enumerate(homes[:-1]):
        for target in homes[i+1:]:
            compute_dijkstra(source, target)
    rough_vertices = get_rough_vertices(lattice, len(homes))
    if rough_vertices:
        # Connect rough vertices to each other
        add_ancilla_nodes(reduced_graph, rough_vertices)
        # Connect rough vertices to rest of graph
        xs, ys = lattice.grid[:, lattice.sites == False]
        for rv in rough_vertices:
            for b in zip(xs, ys):
                lattice_graph.add_edge(rv, b, weight=ANCILLA_WEIGHT)
            # Calculate shortest paths from rough vertices to syndrome sites
            for target in homes:
                compute_dijkstra(rv, target)
    return reduced_graph, paths


def add_ancilla_nodes(graph, rough_vertices):
    ancilla = generate_rough_cluster(rough_vertices)
    graph.add_edges_from(ancilla.edges)


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


def p_to_dist(p):
    return np.log((1-p) / p)
