import numpy as np
import matplotlib.pyplot as plt

from planar_code import PlanarLattice
from graph_operations import *

PRIMAL_EDGES = {'color': 'black', 'linestyle': 'solid', 'linewidth': 1, 'zorder': 1}
PRIMAL_SITES = {'color': 'black', 's': 50, 'marker': 'o', 'zorder': 1}
PRIMAL_ERRORS = {'color': 'red', 'linestyle': 'solid', 'linewidth': 3, 'zorder': 2}
PRIMAL_SYNDROME = {'color': 'orange', 's': 300, 'marker': '+', 'linewidths': 5, 'zorder': 3}

DUAL_EDGES = {'color': 'blue', 'linestyle': 'dashed', 'linewidth': 1, 'zorder': 1}
DUAL_SITES = {'color': 'blue', 's': 50, 'marker': 's', 'zorder': 1}
DUAL_ERRORS = {'color': 'red', 'linestyle': 'dashed', 'linewidth': 3, 'zorder': 2}
DUAL_SYNDROME = {'color': 'orange', 's': 300, 'marker': '+', 'linewidths': 5, 'zorder': 3}

def plot_planar_code(code, plot_dual=True, show_errors=False, show_syndrome=False, save_as=None):
    '''Plot visualization of planar code

    Arguments:
    code -- PlanarCode object

    Keyword arguments:
    plot_dual -- whether or not to plot the dual lattice (default: True)
    show_errors -- whether or not to indicate edges where errors have occurred (default: False)
    show_syndrome -- whether or not to plot (nontrivial) error syndromes (default: False)
    save_as -- if provided, will save plot as the provided file path (default: None)
    '''
    primal = code.primal
    shape = primal.shape
    
    fig, ax = plt.subplots(
        1, 1, figsize=[
            asymptotic_length_scale(i) for i in shape]
    )
    ax.set_aspect('equal')
    ax.grid(False)
    plot_sites(ax, primal, **PRIMAL_SITES)
    plot_edges(
        ax, primal, errors=show_errors,
        edge_style=PRIMAL_EDGES, error_style=PRIMAL_ERRORS
    )
    if show_syndrome:
        primal_syndrome, dual_syndrome = code.measure_syndrome()
        plot_syndrome(ax, primal, primal_syndrome, **PRIMAL_SYNDROME)
    
    if plot_dual:
        boundaries = primal.boundaries
        dual = code.dual
        x_offset = 0.5 * (1 if boundaries[0] != PlanarLattice.SMOOTH else -1)
        y_offset = 0.5 * (1 if boundaries[1] != PlanarLattice.SMOOTH else -1)
        plot_sites(
            ax, dual, x_offset=x_offset, y_offset=y_offset, **DUAL_SITES)
        plot_edges(
            ax, dual, errors=show_errors,
            x_offset=x_offset, y_offset=y_offset,
            edge_style=DUAL_EDGES, error_style=DUAL_ERRORS
        )
        if show_syndrome:
            plot_syndrome(
                ax, dual, dual_syndrome,
                x_offset=x_offset, y_offset=y_offset,
                **DUAL_SYNDROME
            )

    if save_as:
        plt.savefig(save_as)

    
def plot_sites(ax, lattice, x_offset=0, y_offset=0, **kwargs):
    x_coords, y_coords = lattice.grid[:, lattice.sites]
    ax.scatter(x_coords+x_offset, y_coords+y_offset, **kwargs)
    

def plot_edges(
    ax, lattice, x_offset=0, y_offset=0, errors=False,
    edge_style={}, error_style={}
):
    def plot_edge(s0, s1, style):
        try:
            L, W = lattice.shape
            s0x, s0y = s0
            s1x, s1y = s1
            arrow_args = [s0x + x_offset, s0y + y_offset]
            right = (
                s0x == L-1 and s0x > s1x
                and lattice.boundaries[0] == PlanarLattice.PERIODIC
                and lattice.sites[s0]
            )
            upper = (
                s0y == W-1 and s0y > s1y
                and lattice.boundaries[1] == PlanarLattice.PERIODIC
                and lattice.sites[s0]
            )
            if right and upper:
                if s0x > s1x:
                    arrow_args += [1, 0]
                else:
                    arrow_args += [0, 1]
            elif right:
                arrow_args += [1, 0]
            elif upper:
                arrow_args += [0, 1]
            if len(arrow_args) == 4:
                ax.arrow(
                    *arrow_args, length_includes_head=True, width=0, head_width=0.1, **style)
            elif (lattice.sites[s0] or lattice.sites[s1]) and s1x < L and s1y < W:
                ax.plot(
                    np.array((s0x,s1x)) + x_offset,
                    np.array((s0y,s1y)) + y_offset,
                    **style
                )
        except IndexError: pass
        
    L, W, D = lattice.edges.shape
    for x in range(L):
        for y in range(W):
            for d in range(D):
                s0 = lattice.edge_endpoints[:, x, y, d, 0]
                s1 = lattice.edge_endpoints[:, x, y, d, 1]
                if errors and lattice.edges[x, y, d]:
                    style = error_style
                else:
                    style = edge_style
                plot_edge(tuple(s0), tuple(s1), style)


def plot_syndrome(ax, lattice, syndrome, x_offset=0, y_offset=0, **kwargs):
    syndrome_x = lattice.grid[:, syndrome][0] + x_offset
    syndrome_y = lattice.grid[:, syndrome][1] + y_offset
    ax.scatter(syndrome_x, syndrome_y, **kwargs)
    

def asymptotic_length_scale(x, base=1, maximum=50):
    if x > maximum:
        x = maximum * (1 - np.exp(-base * (x + 1) / maximum))
    return x


def plot_matchings(lattice, syndrome, pathfinding=None):
    '''Plot syndrome matching pairs'''
    L, W = lattice.shape
    fig, ax = plt.subplots(1, 1, figsize=(L,L))
    plot_sites(ax, lattice, **PRIMAL_SITES)
    plot_edges(
        ax, lattice, errors=False,
        edge_style=PRIMAL_EDGES, error_style=PRIMAL_ERRORS
    )
    plot_syndrome(ax, lattice, syndrome, **PRIMAL_SYNDROME)

    matching, paths = min_weight_syndrome_matching(lattice, syndrome, pathfinding)
    for i, pair in enumerate(matching):
        for coord in pair:
            if is_real_site(lattice, coord):
                x, y = coord
                ax.text(x, y, s=str(i), c='green', fontsize=20, zorder=10)
        path = paths[i]
        s0x, s0y = path[0]
        for s1x, s1y in path[1:]:
            ax.plot((s0x, s1x), (s0y, s1y), c='green', linewidth=5, zorder=9)
            s0x = s1x
            s0y = s1y
