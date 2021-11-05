import numpy as np
from collections import deque
from planar_code import PlanarCode, PlanarLattice

def lattice_to_edge_list(lattice):
    '''Represent a lattice as a graph (list of edges)
    
    Arguments:
    lattice -- PlanarLattice instance

    Returns dictonary of distances between adjacent vertices in translated graph
    of the format {v0: {v1: edge weight}}
    '''
    L, W = lattice.shape
    X, Y, D = lattice.edges.shape
    adjacency = {}
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
                v0 = gen_vkey(v0x, v0y, L)
                v1 = gen_vkey(v1x, v1y, L)
                if v0_is_site or v1_is_site:
                    weight = lattice.p[x, y, d]
                    set_weight(adjacency, v0, v1, weight)
    return adjacency


def syndrome_to_edge_list(lattice, syndrome, force_manhattan=False):
    '''Represent a syndrome as a fully connected graph (list of edges)

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

    Returns dictonary of distances between adjacent vertices in translated graph
    of the format {v0: {v1: edge weight}}
    '''
    L, W = lattice.shape
    xs, ys = lattice.grid[:, syndrome]
    syndrome_coords = list(zip(xs, ys))
    # Use Manhattan distance if error probabilities are uniform
    if lattice.uniform_p is not None or force_manhattan:
        adjacency = manhattan_graph(lattice, syndrome_coords)
    # Otherwise, use Djikstra's algorithm
    else:
        syndrome_labels = [gen_vkey(*v, L) for v in syndrome_coords]
        adjacency = djikstra_graph(lattice, syndrome_labels)
    return adjacency


def manhattan_graph(lattice, coords):
    '''Generate a fully connected graph between syndrome coordinates

    Assumes uniform error probability and uses Manhattan distance
    to scale edge weights.
    '''
    L, W = lattice.shape
    adjacency = {}
    for i, v0 in enumerate(coords[:-1]):
        k0 = gen_vkey(*v0, L)
        for v1 in coords[i+1:]:
            k1 = gen_vkey(*v1, L)
            weight = manhattan_distance(lattice, v0, v1)
            set_weight(adjacency, k0, k1, weight)
    rough_vertices = get_rough_labels(lattice, len(coords))
    if rough_vertices:
        # Connect rough vertices to each other
        adjacency.update(generate_rough_cluster(rough_vertices))
        # Connect rough vertices to rest of vertices
        for rv in rough_vertices:
            for v0 in coords:
                weight = manhattan_dist_to_rough(lattice, v0)
                v = gen_vkey(*v0, L)
                set_weight(adjacency, rv, v, weight)
    return adjacency


def manhattan_distance(lattice, v0, v1):
    deltas = []
    for d in range(PlanarLattice.D):
        delta = abs(v1[d] - v0[d])
        # Handle periodic boundaries
        if lattice.boundaries[2*d] == PlanarLattice.PERIODIC:
            delta = min(
                delta,
                v0[d] + lattice.shape[d]-1-v1[d],
                v1[d] + lattice.shape[d]-1-v0[d]
            )
        deltas.append(delta)
    return sum(deltas)


def manhattan_dist_to_rough(lattice, v0):
    L, W = lattice.shape
    x, y = v0
    left, right, bottom, top = (lattice.boundaries == PlanarLattice.ROUGH)
    deltas = []
    if left: deltas.append(x)
    if right: deltas.append(L-1-x)
    if bottom: deltas.append(y)
    if top: deltas.append(W-1-y)
    return min(deltas)


def djikstra_graph(lattice, homes):
    '''Generate a fully connected graph between syndrome coordinates

    Uses Djikstra's algorithm to find the minimum distance between all coords marked home.
    '''
    L, W = lattice.shape
    neighbors = lattice_to_edge_list(lattice)
    remaining_homes = list(homes)
    adjacency = {}
    while len(remaining_homes) > 1:
        adjacency.update(djikstra_paths(lattice, remaining_homes, neighbors))
        del remaining_homes[0]
    rough_vertices = get_rough_labels(lattice, len(homes))
    if rough_vertices:
        # Connect rough vertices to each other
        adjacency.update(generate_rough_cluster(rough_vertices))
        # Connect rough vertices to rest of graph
        xs, ys = lattice.grid[:, lattice.sites == False]
        rough_boundary = [gen_vkey(*b, L) for b in zip(xs, ys)]
        for rv in rough_vertices:
            for b in rough_boundary:
                set_weight(neighbors, rv, b, 0.5)
        # Calculate shortest paths from rough vertices to syndrome sites
        while len(rough_vertices) > 0:
            rv = rough_vertices[0]
            rough_homes = [rv] + homes
            rv_neighbors = djikstra_paths(lattice, rough_homes, neighbors)[rv]
            for home, dist in rv_neighbors.items():
                set_weight(adjacency, rv, home, dist)
            del rough_vertices[0]
    return adjacency


def djikstra_paths(lattice, homes, neighbors):
    L, W = lattice.shape
    home = homes[0]
    distances = {v: np.inf for v in range(L*W)}
    unvisited = list(distances.keys())
    distances[home] = -np.inf
    while len(unvisited) > 0:
        k0 = min(unvisited, key=lambda i: distances[i])
        del unvisited[unvisited.index(k0)]
        for k1, p in neighbors[k0].items():
            dist = p_to_dist(p)
            try:
                old_dist = distances[k1]
            except KeyError:
                continue
            this_dist = distances[k0] + dist
            if this_dist < old_dist:
                distances[k1] = this_dist
    adjacency = {}
    for destination in homes[1:]:
        set_weight(adjacency, home, destination, distances[destination])
    return adjacency


def generate_rough_cluster(vertex_labels):
    cluster = {}
    n = len(vertex_labels)
    for i in range(n-1):
        k0 = vertex_labels[i]
        for j in range(i+1, n):
            k1 = vertex_labels[j]
            set_weight(cluster, k0, k1, 0)
    return cluster


def get_rough_labels(lattice, how_many):
    if np.any(lattice.boundaries == PlanarLattice.ROUGH):
        L, W = lattice.shape
        return [i+L*W for i in range(how_many)]
    return None


def set_weight(map, v0, v1, weight):
    for v in (v0, v1):
        if v not in map:
            map[v] = {}
    map[v0][v1] = weight
    map[v1][v0] = weight


def p_to_dist(p):
    return np.log((1-p) / p)


def gen_vkey(vx, vy, L):
    return L*vy + vx


def unpack_vkey(key, L):
    vx = key % L
    vy = key // L
    return vx, vy
