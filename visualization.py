import numpy as np
import matplotlib.pyplot as plt

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
    plot_lattice(ax, primal, show_errors=show_errors, show_syndrome=show_syndrome)
    
    if plot_dual:
        boundaries = primal.boundaries
        dual = code.dual
        x_offset = 0.5 * (1 if ~boundaries[0] else -1)
        y_offset = 0.5 * (1 if ~boundaries[1] else -1)
        plot_lattice(
            ax, dual, show_errors=show_errors, show_syndrome=show_syndrome,
            x_offset=x_offset, y_offset=y_offset,
            site_style=DUAL_SITES, edge_style=DUAL_EDGES,
            error_style=DUAL_ERRORS, syndrome_style=DUAL_SYNDROME
        )

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')


def plot_lattice(
    ax, lattice, show_errors=True, show_syndrome=True,
    x_offset=0, y_offset=0,
    site_style=PRIMAL_SITES, edge_style=PRIMAL_EDGES,
    error_style=PRIMAL_ERRORS, syndrome_style=PRIMAL_SYNDROME
):
    plot_sites(
        ax, lattice, x_offset=x_offset, y_offset=y_offset, **site_style)
    plot_edges(
        ax, lattice, errors=show_errors,
        x_offset=x_offset, y_offset=y_offset,
        edge_style=edge_style, error_style=error_style
    )
    if show_syndrome:
        try:
            syndrome = lattice.measure_syndrome(debug=True)
        except TypeError:
            syndrome = lattice.measure_syndrome()
        plot_syndrome(
            ax, lattice, syndrome,
            x_offset=x_offset, y_offset=y_offset,
            **syndrome_style
        )

    
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
            if (lattice.sites[s0] or lattice.sites[s1]) and s1x < L and s1y < W:
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
    if errors and hasattr(lattice, 'correlated_syndromes'):
        for sites, error in lattice.correlated_syndromes.items():
            if error:
                plot_edge(*sites, error_style)


def plot_syndrome(ax, lattice, syndrome, x_offset=0, y_offset=0, **kwargs):
    if type(syndrome) == tuple:
        syndrome = syndrome[0]
    syndrome_x = lattice.grid[:, syndrome][0] + x_offset
    syndrome_y = lattice.grid[:, syndrome][1] + y_offset
    ax.scatter(syndrome_x, syndrome_y, **kwargs)
    

def asymptotic_length_scale(x, base=1, maximum=50):
    if x > maximum:
        x = maximum * (1 - np.exp(-base * (x + 1) / maximum))
    return x


def plot_matchings(lattice, syndrome, matching, paths):
    '''Plot syndrome matching pairs'''
    L, W = lattice.shape
    fig, ax = plt.subplots(1, 1, figsize=(L,L))
    plot_sites(ax, lattice, **PRIMAL_SITES)
    plot_edges(
        ax, lattice, errors=False,
        edge_style=PRIMAL_EDGES, error_style=PRIMAL_ERRORS
    )
    plot_syndrome(ax, lattice, syndrome, **PRIMAL_SYNDROME)
    for i, pair in enumerate(matching):
        for coord in pair:
            if lattice.is_real_site(coord):
                x, y = coord
                ax.text(x, y, s=str(i), c='green', fontsize=20, zorder=10)
        path = paths[i]
        s0x, s0y = path[0]
        for s1x, s1y in path[1:]:
            ax.plot((s0x, s1x), (s0y, s1y), c='green', linewidth=5, zorder=9)
            s0x = s1x
            s0y = s1y


def plot_error_probabilities(lattice, verbose=True, syndrome=None, save_as=None):
    '''Plot edges and associated error probabilities'''
    fig, ax = plt.subplots(
        1, 1, figsize=[
            asymptotic_length_scale(i) for i in lattice.shape]
    )
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_frame_on(False)
    plot_sites(ax, lattice, **PRIMAL_SITES)

    graph = lattice.to_graph()
    for s0, s1, weight in graph.edges.data(WEIGHT_KEY):
        s0x, s0y = s0
        s1x, s1y = s1
        ax.plot((s0x, s1x), (s0y, s1y), **PRIMAL_EDGES)
        if verbose:
            rot = 315 if (s0x != s1x) and (s0y != s1y) else 0
            ax.text(
                s0x+(s1x-s0x)/2, s0y+(s1y-s0y)/2, str(round(weight, 4)),
                c='blue', backgroundcolor='white',
                bbox=dict(facecolor='white', edgecolor='black'),
                weight='bold', ha='center', va='center',
                rotation=rot, zorder=10
            )
    
    if syndrome is not None:
        plot_syndrome(ax, lattice, syndrome, **PRIMAL_SYNDROME)

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
