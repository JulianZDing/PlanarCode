import networkx as nx
import numpy as np

DEFAULT_BOUNDS = np.array((1,0), dtype=bool)

class OperatorException(Exception): pass
class LogicalException(Exception): pass


class PlanarLattice:

    def __init__(self, L, W, boundaries, p):
        '''Class definition of a rectangular planar lattice.
        
        Only defines edges and sites (no plaquettes); use a second object for dual lattice.
        NOTE: For sanity...
            - "length" and "horizontal" refers to x dimension
            - "width" and "vertical" refers to y dimension
            - x corresponds to array index at position 0 (rows)
            - y corresponds to array index at position 1 (columns)
        
        Arguments:
        L -- length of lattice in number of sites (includes missing sites on rough boundaries)
        W -- width of lattice in number of sites (includes missing sites on rough boundaries)
        boundaries -- boolean array for smoothness of boundaries (x, y)
                      - 0: rough boundary
                      - 1: smooth boundary
        p -- error probability for each edge operator
             (either single value, or array with same shape as lattice edges)
        '''
        if L < 2 or W < 2:
            raise LogicalException('Lattices with site length/width < 2 have no plaquettes.')

        self.shape = np.array((L, W))
        self.grid = np.indices(self.shape)
        self.boundaries = np.array(boundaries, dtype=bool)
        both_smooth = np.all(self.boundaries)
        both_rough = np.all(~self.boundaries)
        if both_smooth or both_rough:
            raise LogicalException(
                'Lattices with all rough or all smooth boundaries cannot encode a qubit.')
        
        # [x, y, 0] for horizontal edge, [x, y, 1] for vertical edge
        # edges with active errors have True values
        self.edges = np.zeros((L, W, self.shape.size), dtype=bool)
        self.edge_endpoints = self._endpoint_coords()
        self._set_lattice_sites()
        
        self.uniform_p = None
        if not isinstance(p, np.ndarray):
            self.uniform_p = p
            p = np.full(self.edges.shape, p)
        elif p.shape != self.edges.shape:
            raise IndexError(
                f'Error probability array of shape {p.shape} does not match a lattice with edge dimensions {self.edges.shape}')
        self.p = p
        self.original_p = np.copy(p)

    
    def to_graph(self, error_model=None, p_scaling_fn=None):
        '''Represent a lattice as a networkx Graph
        with edges weighted according to error probability

        Keyword arguments:
        error_model -- array of error probabilities matching self.edges.shape
                       (if unspecified, uses self.p)
        p_scaling_fn -- optional function to transform probabilities

        Returns nx.Graph instance
        '''
        if error_model is None:
            error_model = self.p
        X, Y, D = self.edges.shape
        graph = nx.Graph()
        for x in range(X):
            for y in range(Y):
                for d in range(D):
                    vx, vy = self.edge_endpoints[:, x, y, d]
                    v0x, v1x = vx
                    v0y, v1y = vy
                    v0 = (v0x, v0y)
                    v1 = (v1x, v1y)
                    if not self.is_real_site(v1):
                        continue
                    v0_is_site = self.sites[v0]
                    v1_is_site = self.sites[v1]
                    if v0_is_site or v1_is_site:
                        p = error_model[x, y, d]
                        weight = p if p_scaling_fn is None else p_scaling_fn(p)
                        graph.add_edge(v0, v1, weight=weight)
        return graph

    
    def is_real_site(self, v):
        '''Determines if a coordinate v = (x, y) is a valid site within the lattice'''
        L, W = self.shape
        x, y = v
        return x < L and y < W

    
    def is_real_edge(self, coords):
        '''Determines if a coordinate (x, y, d) is a valid edge within the lattice'''
        x, y, d = coords
        try:
            vx, vy = self.edge_endpoints[:, x, y, d]
            s0 = (vx[0], vy[0])
            s1 = (vx[1], vy[1])
            return self.is_real_site(s0) and self.is_real_site(s1)
        except IndexError:
            return False


    def apply_edge_operators(self, targets):
        '''Apply edge operators

        Arguments:
        targets -- list of site coordinates or edge coordinates
                   If target is a string of site coordinates,
                   the sites must be next to each other
        '''
        if len(targets) == 0:
            return
        length = len(targets[0])
        if length == len(self.shape):
            self._apply_edge_operators_sites(targets)
        elif length == len(self.edges.shape):
            self._apply_edge_operators_edges(targets)
        else:
            raise OperatorException(
                f'Coordinates of length {length} do not match dimensions of sites or edges.')
    
    
    def apply_edge_operator(self, select):
        '''Apply edge operators identified by a coordinate tuple or index/boolean array'''
        self.edges[select] = ~self.edges[select]
        if self.rough_edge_mask is not None:
            self.edges[self.rough_edge_mask] = False
    

    def _apply_edge_operators_sites(self, sites):
        if len(sites) < 2:
            raise OperatorException(
                '1 site coordinate does not adequately identify an edge.')
        edge_list = []
        prev = sites[0]
        for curr in sites[1:]:
            deltas = np.array(curr) - np.array(prev)
            if np.sum(np.abs(deltas)) != 1:
                raise OperatorException(
                    f'Sites {prev} and {curr} are not adjacent.')
            d = np.argmax(np.abs(deltas))
            edge_list.append((*prev, d) if deltas[d] > 0 else (*curr, d))
            prev = curr
        self._apply_edge_operators_edges(edge_list)


    def _apply_edge_operators_edges(self, edges):
        for edge in edges:
            self.apply_edge_operator(edge)


    def apply_errors(self):
        '''Apply errors on edge operators according to self.p'''
        probs = np.random.rand(*self.edges.shape)
        errors = self.p > probs
        self.apply_edge_operator(errors)

    
    def measure_syndrome(self):
        '''Measure the error syndrome 
        
        Nontrivial syndrome measurements are represented by True values.

        Returns array of syndrome corresponding to self.shape
        '''
        L, W = self.shape
        syndrome = np.zeros(self.shape, dtype=int)
        # Quasi-particle pair creation
        xs, ys = self.edge_endpoints[:, self.edges]
        valid = None
        for d in range(self.shape.size):
            this_valid = (xs[:,d] < L) & (ys[:,d] < W)
            if valid is None:
                valid = this_valid
            else:
                valid = valid & this_valid
        for x, y in zip(xs[valid].flatten(), ys[valid].flatten()):
            syndrome[x, y] += 1
        # Quasi-particle annihilation
        syndrome = np.array(syndrome % 2, dtype=bool)
        # Remove rough boundary missing sites
        syndrome[~self.sites] = False
        return syndrome


    def detect_logical_errors(self, initial_z=False):
        '''Detect whether or not a logical error has occurred

        Checks number of intersections of logical X with edge operators (modulo 2).

        Keyword arguments:
        initial_z -- whether a logical Z was already applied to the lattice (default: False)

        Returns if a logical error has occurred
        '''
        if np.sum(self.measure_syndrome()) > 0:
            raise LogicalException(
                'Non-trivial syndrome detected; lattice is not in the codespace.')
        d = np.argmin(self.boundaries)
        check_row = self.shape[d] // 2
        indices = {d: check_row, (self.edges.ndim-1): d}
        ix = tuple([indices.get(i, slice(None)) for i in range(self.edges.ndim)])
        intersections = np.sum(self.edges[ix])
        return (intersections % 2) != initial_z

    
    def reset(self):
        '''Reset the error state of the lattice'''
        self.edges = np.zeros(self.edges.shape, dtype=bool)
        self.p = np.copy(self.original_p)
        
    
    def _set_lattice_sites(self):
        '''Set planar code sites and edge mask based on roughness of boundaries
        
        A True site means a site is present
        A False site denotes a hole or part of a rough boundary
        '''
        self.sites = np.ones(self.shape, dtype=bool)
        # Set smoothness of boundaries
        select = slice(None)
        ends = [0,-1]
        self.rough_edge_mask = None
        for i, smooth in enumerate(self.boundaries):
            if not smooth:
                dim_range = range(self.shape.size)
                indices = {
                    i: ends,
                    (self.edges.ndim-1): [j for j in dim_range if j != i]
                }
                ix = tuple([indices.get(j, select) for j in dim_range])
                self.sites[ix] = False
                self.rough_edge_mask = tuple([
                    indices.get(j, select) for j in range(self.edges.ndim)])


    def _endpoint_coords(self):
        '''Generate an array with coordinates of endpoint sites for all edges

        Entry (x, y, d) corresponds to the two sites [[s0x, s0y], [s1x, s1y]]
        touching the edge of orientation d rooted at position (x, y) on the lattice
        '''
        L, W, D = self.edges.shape
        endpoints = []
        for d in range(D):
            start = self.grid
            end = np.copy(self.grid)
            end[d] += 1
            endpoints.append(np.stack((start, end), axis=-1))
        return np.stack(endpoints, axis=-2)


class PlanarCode:
    
    def __init__(self, L, W=None, boundaries=DEFAULT_BOUNDS, pz=0.05, px=None):
        '''Class definition of a planar code.
        
        Defines edge and site operators on primal lattice, plaquette operators on dual lattice.
        
        Arguments:
        L -- side length of planar code (in number of sites)
        
        Keyword arguments:
        W -- width of planar code, if different from length (default: same as length)
        boundaries -- boolean array for smoothness of boundaries (x, y)
                      - 0: rough boundary
                      - 1: smooth boundary
        pz -- Z error probability for each physical qubit (edge operator)
             (default: 0.05)
        px -- X error probability for each physical qubit (plaquette edge operator)
                  (default: same as pz)
        '''
        W = L if W is None else W
        self.primal = PlanarLattice(L, W, boundaries, pz)
        
        boundaries_dual = ~boundaries
        L_dual = L - np.sum(boundaries_dual[0]) + np.sum(~boundaries_dual[0])
        W_dual = W - np.sum(boundaries_dual[1]) + np.sum(~boundaries_dual[1])
        px = pz if px is None else px
        self.dual = PlanarLattice(L_dual, W_dual, boundaries_dual, px)
        
        self.lattices = (self.primal, self.dual)
        self.tick = 0
    

    def primal_to_dual(self, coords, reverse=False):
        '''Translate edge coordinates to coordinates of intersecting edge on dual lattice

        Arguments:
        coords --- edge coordinates (x, y, d)

        Keyword arguments:
        reversed -- translate from dual to primal instead

        Returns the coordinates of the intersecting edge on the dual lattice (x', y', d')
        '''
        if len(coords) != 3:
            raise ValueError(f'{coords} is not a valid edge coordinate')

        x_smooth = self.primal.boundaries[0]
        x, y, d = coords
        if (
            (x_smooth and d == 0 and not reverse)
            or (not x_smooth and d == 0 and reverse)
        ):
            x += 1
            y -= 1
        elif (
            (not x_smooth and d == 1 and not reverse)
            or (x_smooth and d == 1 and reverse)
        ):
            x -= 1
            y += 1
        
        d = (d+1) % 2

        return (x, y, d)


    def advance(self, ticks=1):
        '''Apply stochastic errors on primal and dual lattices
        
        Arguments:
        ticks -- number of time steps to advance
        '''
        for i in range(ticks):
            for lattice in self.lattices:
                lattice.apply_errors()
            self.tick += 1
    
    
    def measure_syndrome(self, advance=0, **kwargs):
        '''Measure the error syndrome at the current time
        
        Nontrivial syndromes on the primal and dual lattices are represented by True values.

        Keyword arguments:
        advance -- advance time by this many ticks after measurement (default: 0)

        Returns arrays of error syndromes for both primal and dual lattices
        '''
        syndromes = []
        for lattice in self.lattices:
            try:
                syndromes.append(lattice.measure_syndrome(**kwargs))
            except:
                syndromes.append(lattice.measure_syndrome())
        if advance > 0:
            self.advance(advance)
        return syndromes


    def reset(self):
        '''Reset the error state of both primal and dual lattices'''
        for lattice in self.lattices:
            lattice.reset()
