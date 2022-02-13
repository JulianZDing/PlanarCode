from collections import defaultdict
from planar_code import *

class CorrelatedPlanarLattice(PlanarLattice):
    
    def __init__(self, r, *args, **kwargs):
        '''Class definition for a rectangular planar lattice where syndrome measurements can fail

        Arguments:
        r -- probability of photon loss in waveguide at each site
             (either single value, or array with same shape as lattice)

        Rest of arguments are passed to PlanarLattice constructor.
        '''
        super().__init__(*args, **kwargs)
        self.uniform_p = None
        if not isinstance(r, np.ndarray):
            r = np.full(self.shape, r)
        elif r.shape != self.shape:
            raise IndexError(
                f'Waveguide failure probability array of shape {r.shape} does not match a lattice with site dimensions {self.shape}')
        self.r = r
        self.p_fail = 1 - (1-r)**5
        self.reset_error_model()

    
    @staticmethod
    def dump_error(target, key, q):
        '''
        Increase the probability of error for an element of a collection

        Arguments:
        target -- dictionary containing error probabilities
        key -- identifier for specific entry in target
        q -- amount of error to dump onto target[key]

        Changes target[key] in place
        '''
        p = target[key]
        target[key] = (1-p)*q + p*(1-p)

    
    def to_graph(self, p_scaling_fn=None, naive=False):
        '''Represent a lattice as a networkx Graph
        with edges weighted according to error probability.

        Add extra edges to represent correlated errors.

        Keyword arguments:
        p_scaling_fn -- optional function to transform probabilities
        naive -- whether or not to use the heralded error model

        Returns nx.Graph instance
        '''
        error_model = self.p if naive else self.heralded_p
        graph = super().to_graph(error_model, p_scaling_fn)
        if not naive:
            for (v0, v1), p in self.correlated_syndrome_p.items():
                weight = p if p_scaling_fn is None else p_scaling_fn(p)
                graph.add_edge(v0, v1, weight=weight)
        return graph

    
    def _apply_edge_operators_sites(self, sites):
        if len(sites) < 2:
            raise OperatorException(
                '1 site coordinate does not adequately identify an edge.')
        edge_list = []
        prev = sites[0]
        for curr in sites[1:]:
            deltas = np.array(curr) - np.array(prev)
            abs_deltas = np.abs(deltas)
            sum_deltas = np.sum(abs_deltas)
            if np.any(abs_deltas > 1) or sum_deltas < 1:
                raise OperatorException(
                    f'Sites {prev} and {curr} are not adjacent.')
            if sum_deltas > 1:
                self.apply_z1z2(prev, curr)
            else:
                d = np.argmax(abs_deltas)
                edge_list.append((*prev, d) if deltas[d] > 0 else (*curr, d))
            prev = curr
        self._apply_edge_operators_edges(edge_list)
    

    def apply_z1z2(self, s0, s1):
        '''Apply a Z1Z2 operation (correlated syndromes)'''
        s0x, s0y = s0
        s1x, s1y = s1
        dx = s1x - s0x
        dy = s1y - s0y
        v0x = s0x if dx > 0 else s1x
        v1x = s1x
        v0y, v1y = (s0y, s0y) if dy > 0 else (s0y, s1y)
        self._apply_edge_operators_edges([(v0x, v0y, 0), (v1x, v1y, 1)])
    

    def measure_syndrome(self, failed_sites=None, debug=False):
        '''
        Measure the error syndrome
        
        Takes into account correlated errors and possibility of measurement failure
        Repeats syndrome measurements until success is achieved at all sites
        
        Keyword arguments:
        failed -- coordinates/mask for syndrome measurements that will fail
        debug -- prevents failed measurements
        '''
        syndrome = super().measure_syndrome()

        if debug:
            return syndrome

        if failed_sites is None:
            failed_sites = np.random.rand(*self.shape) < self.p_fail
        xs, ys = self.grid[:, failed_sites]
        retries = {}
        for x, y in zip(xs,ys):
            retries[(x, y)] = self.measure_syndrome_again(x, y)
        
        return syndrome, retries
    

    def measure_syndrome_again(self, x, y, tries=1):
        '''
        Repeat measurement of error syndrome at the site x, y.
        Can fail again according to probability of measurement failure.

        Arguments:
        x -- site x coordinate
        y -- site y coordinate

        Returns number of tries before success
        '''
        failed = np.random.rand() < self.p_fail[x, y]
        if failed:
            return self.measure_syndrome_again(x, y, tries=tries+1)
        return tries

    
    def detect_logical_errors(self, initial_z=False):
        '''Detect whether or not a logical error has occurred

        Checks number of intersections of logical X with edge operators (modulo 2).

        Keyword arguments:
        initial_z -- whether a logical Z was already applied to the lattice (default: False)

        Returns if a logical error has occurred
        '''
        if np.sum(self.measure_syndrome(debug=True)) > 0:
            raise LogicalException(
                'Non-trivial syndrome detected; lattice is not in the codespace.')
        return super()._detect_logical_errors(initial_z)


    def reset_error_model(self):
        '''Reset the error model'''
        self.correlated_syndrome_p = defaultdict(float)
        self.heralded_p = np.copy(self.p)


class PhotonicPlanarCode(PlanarCode):

    def __init__(self, L, W=None, boundaries=DEFAULT_BOUNDS, pz=0, px=None, rz=0, rx=None):
        '''Class definition of a planar code implemented using cavities and photonic measurement apparatus.
        
        Arguments:
        L -- side length of planar code (in number of sites)
        
        Keyword arguments:
        W -- width of planar code, if different from length (default: same as length)
        boundaries -- boolean array for smoothness of boundaries (horizontal, vertical)
                      - 0: rough boundary
                      - 1: smooth boundary
        pz -- Z error probability for each physical qubit (edge operator)
              (default: 0.05)
        px -- X error probability for each physical qubit (plaquette edge operator)
              (default: same as pz)
        rz -- Probability for plaquette operator measurements to fail (default: 0)
        rx -- Probability for site operator measurements to fail (default: same as rz)
        '''
        W = L if W is None else W
        self.primal = CorrelatedPlanarLattice(rz, L, W, boundaries, pz)
        
        boundaries_dual = ~boundaries
        L_dual = L - np.sum(boundaries_dual[0]) + np.sum(~boundaries_dual[0])
        W_dual = W - np.sum(boundaries_dual[1]) + np.sum(~boundaries_dual[1])
        px = pz if px is None else px
        rx = rz if rx is None else rx
        self.dual = CorrelatedPlanarLattice(rx, L_dual, W_dual, boundaries_dual, px)
        
        self.lattices = (self.primal, self.dual)
        self.tick = 0


    def apply_failed_measurement_error(self, x, y, reverse=False):
        '''
        Dump errors onto neighboring edge operators when a syndrome measurement fails
        Also apply errors to edges due to measurement failure error model

        Arguments:
        x -- site x coordinate
        y -- site y coordinate

        Keyword arguments:
        reverse -- whether primal and dual labels are reversed
        '''
        i = int(reverse)
        primal = self.lattices[i]
        dual = self.lattices[(i+1) % 2]
        # up right down left
        z = [(x, y, 1), (x, y, 0), (x, y-1, 1), (x-1, y, 0)]
        z = [self.primal_to_dual(coords, reverse) for coords in z]
        r = primal.r[x, y]
        apply_error = [lambda: None for _ in range(5)]

        # Update error model
        p0 = 1 - (1-r)**5
        q1 =  r*(1-r) / (2*p0)
        q12 = r*(1-r)**2 / (2*p0)
        q4 = r*(1-r)**3 / (2*p0)

        for q, zi, ei in [(q1, 0, 1), (q4, 3, 3)]:
            edge = z[zi]
            if dual.is_real_edge(edge):
                CorrelatedPlanarLattice.dump_error(dual.heralded_p, edge, q)
                apply_error[ei] = lambda: dual.apply_edge_operator(edge)

        s0 = z[0][:2]
        s1 = z[1][:2]
        if (
            dual.is_real_site(s0) and dual.is_real_site(s1)
            and dual.sites[s0] and dual.sites[s1]
        ):
            CorrelatedPlanarLattice.dump_error(
                dual.correlated_syndrome_p, (s0, s1), q12)
            apply_error[2] = lambda: dual.apply_z1z2(s0, s1)
        
        # Simulate photon disappearing in waveguide and apply appropriate error
        for i in range(5):
            if np.random.rand() < r and np.random.rand() < 0.5:
                apply_error[i]()
                break


    def measure_syndrome(self, lattice=0, debug=False, **kwargs):
        '''Measure the error syndrome at the selected lattice at the current time.
        Also propagates errors from failed measurements.
        
        Nontrivial syndromes on the primal and dual lattices are represented by True values.

        Returns error syndrome for primal (lattice=0) or dual (lattice=1) lattice
        '''
        out = self.lattices[lattice].measure_syndrome(debug=debug, **kwargs)
        if debug:
            return out
        syndrome, retries = out
        for (x, y), tries in retries.items():
            for _ in range(tries):
                self.apply_failed_measurement_error(x, y, reverse=bool(lattice))
        return syndrome
