import numpy as np

DEFAULT_BOUNDS = np.array([1, 1, 0, 0], dtype=int)

class BoundaryException(Exception): pass

class PlanarCode:
    
    def __init__(self, L, W=None, boundaries=DEFAULT_BOUNDS, p=0, p_dual=None):
        '''Class definition of a square planar code.
        
        Defines edge and site operators on primal lattice, plaquette operators on dual lattice.
        
        Arguments:
        L -- side length of planar code (in number of sites)
        
        Keyword arguments:
        W -- width of planar code, if different from length (default: same as length)
        boundaries -- boolean array for smoothness of (primal lattice) boundaries
                      (left, right, down, up; default: [1, 1, 0, 0])
        p -- error probability for each edge operator (default: 0)
        p_dual -- error probability for each plaquette "edge" operator (default: same as primal_p)
        '''
        W = L if W is None else W
        self.primal = PlanarLattice(L, W, boundaries, p)
        
        boundaries_dual = np.copy(boundaries)
        boundaries_dual[boundaries == PlanarLattice.SMOOTH] = PlanarLattice.ROUGH
        boundaries_dual[boundaries == PlanarLattice.ROUGH] = PlanarLattice.SMOOTH
        L_dual = L - np.sum(boundaries_dual[:2] != PlanarLattice.ROUGH) + 1
        W_dual = W - np.sum(boundaries_dual[2:] != PlanarLattice.ROUGH) + 1
        p_dual = p if p_dual is None else p_dual
        self.dual = PlanarLattice(L_dual, W_dual, boundaries_dual, p_dual)
        
        self.lattices = [self.primal, self.dual]
        self.tick = 0


    def advance(self, ticks=1):
        '''Apply stochastic errors primal and dual lattices
        
        Arguments:
        ticks -- number of time steps to advance
        '''
        for i in range(ticks):
            for lattice in self.lattices:
                lattice.apply_errors()
            self.tick += 1
    
    
    def measure_syndrome(self, advance=0):
        '''Measure the error syndrome at the current time
        
        Nontrivial syndromes on the primal and dual lattices are represented by True values.

        Keyword arguments:
        advance -- advance time by this many ticks after measurement (default: 0)

        Returns arrays of error syndromes for both primal and dual lattices
        '''
        syndromes = [lattice.measure_syndrome() for lattice in self.lattices]
        if advance > 0:
            self.advance(advance)
        return syndromes


    def reset(self):
        '''Reset the error state of both primal and dual lattices'''
        for lattice in self.lattices:
            lattice.reset()


class PlanarLattice:

    ROUGH, SMOOTH, PERIODIC = (0, 1, 2)
    D = 2

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
        boundaries -- integer array for smoothness of boundaries (left, right, down, up)
                      - 0: rough boundary
                      - 1: smooth boundary
                      - 2: periodic boundary
        p -- error probability for each edge operator
             (either single value, or array with same shape as lattice)
        '''
        self.shape = np.array((L, W))
        self.grid = np.indices(self.shape)

        self.boundaries = np.array(boundaries, dtype=int)
        periodic_x = boundaries[:PlanarLattice.D] == PlanarLattice.PERIODIC
        if np.any(periodic_x) and not np.all(periodic_x):
            raise BoundaryException(
                'Periodic condition for left and right boundaries does not match.')
        periodic_y = boundaries[PlanarLattice.D:] == PlanarLattice.PERIODIC
        if np.any(periodic_y) and not np.all(periodic_y):
            raise BoundaryException(
                'Periodic condition for bottom and top boundaries does not match.')
        
        self.sites = self._generate_lattice_sites()
        # [x, y, 0] for horizontal edge, [x, y, 1] for vertical edge
        # edges with active errors have True values
        self.edges = np.zeros((L, W, PlanarLattice.D), dtype=bool)
        self.edge_endpoints = self._endpoint_coords()
        
        self.uniform_p = None
        if not isinstance(p, np.ndarray):
            self.uniform_p = p
            p = np.full(self.edges.shape, p)
        elif p.shape != self.edges.shape:
            raise IndexError(
                f'Error probability array of shape {p.shape} does not match a lattice with edge dimensions {self.edges.shape}')
        self.p = p


    def apply_errors(self):
        '''Apply errors on edge operators according to self.p'''
        probs = np.random.rand(*self.edges.shape)
        errors = self.p > probs
        self.edges[errors] = ~self.edges[errors]

    
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
        for d in range(PlanarLattice.D):
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

    
    def reset(self):
        '''Reset the error state of the lattice'''
        self.edges = np.zeros(self.edges.shape, dtype=bool)
        
    
    def _generate_lattice_sites(self):
        '''Generate a boolean array of planar code sites
        
        A True site means a site is present
        A False site denotes a hole or part of a rough boundary
        '''
        sites = np.ones(self.shape, dtype=bool)
        # Set smoothness of boundaries
        select_x = np.ones(sites.shape[0], dtype=bool)
        select_y = np.ones(sites.shape[1], dtype=bool)
        wall_masks = [(0, select_y), (-1, select_y), (select_x, 0), (select_x, -1)]
        for i, b in enumerate(self.boundaries):
            if b == PlanarLattice.ROUGH:
                sites[wall_masks[i]] = False
        return sites

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
            if self.boundaries[2*d] == PlanarLattice.PERIODIC:
                end[d] = (end[d] + 1) % self.shape[d]
            else:
                end[d] += 1
            endpoints.append(np.stack((start, end), axis=-1))
        return np.stack(endpoints, axis=-2)
