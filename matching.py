import numpy as np
from scipy.optimize import linear_sum_assignment
from graph_operations import *


class MatchingException(Exception): pass


def dict_to_matrix_translation(adjacency):
    '''Translate an adjacency dictionary to a cost matrix representation

    Arguments:
    adjacency -- dictionary of the form {v0: {v1: edge weight}}

    Returns:
        - nxn array C where C[i,j] corresponds to the distance
          between vertex i and vertex j
        - 1xn array L where L[i] corresponds to the original label
          for the vertex corresponding to row/column i in C
    '''
    n = len(adjacency)
    index_to_vertex = list(enumerate(adjacency.keys()))
    cost = np.full((n,n), np.inf)
    for i, v0 in index_to_vertex[:-1]:
        for j, v1 in index_to_vertex[i+1:]:
            dist = adjacency[v0][v1]
            cost[i,j] = dist 
            cost[j,i] = dist
    return cost, [v for i, v in index_to_vertex]

def min_weight_matching(lattice, syndrome, force_manhattan=False):
    '''Find a minimum weight perfect matching of syndromes on a lattice

    Arguments:
    lattice -- PlanarLattice instance
    syndrome -- lattice.shape boolean array marking nontrivial syndromes

    Positional arguments:
    force_manhattan -- use Manhattan distance calculation regardless of error model

    Returns list of paired site coordinates (None corresponds to a rough edge)
    '''
    L, W = lattice.shape
    graph = syndrome_to_edge_list(
        lattice, syndrome, force_manhattan=force_manhattan)
    cost, vkeys = dict_to_matrix_translation(graph)
    matching = list(zip(*linear_sum_assignment(cost)))
    matching = set(tuple(sorted(pair)) for pair in matching)
    matched_coords = []
    for pair in matching:
        coords = []
        for k in pair:
            key = vkeys[k]
            if key < L*W:
                coords.append(unpack_vkey(key, L))
        if coords:
            if len(coords) < 2:
                coords.append(None)
            matched_coords.append(coords)
    return matched_coords
