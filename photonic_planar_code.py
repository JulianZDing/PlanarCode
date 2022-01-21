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
        self.correlated_syndrome_p = defaultdict(float)
        self.correlated_syndromes = defaultdict(bool)

    
    @staticmethod
    def dump_error(target, key, new_p):
        '''
        Increase the probability of error for an element of a collection

        Arguments:
        target -- dictionary containing error probabilities
        key -- identifier for specific entry in target
        new_p -- amount of error to dump onto target[key]

        Changes target[key] in place
        '''
        p = target[key]
        target[key] = 1 - (1-p) * (1-new_p)

    
    def to_graph(self, p_scaling_fn=None, naive=False):
        '''Represent a lattice as a networkx Graph
        with edges weighted according to error probability.

        Add extra edges to represent correlated errors.

        Keyword arguments:
        p_scaling_fn -- optional function to transform probabilities
        naive -- whether or not to use the heralded error model

        Returns nx.Graph instance
        '''
        error_model = self.original_p if naive else None
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
            for key in [(prev, curr), (curr, prev)]:
                if key in self.correlated_syndromes:
                    self.apply_z1z2(*key)
                continue

            deltas = np.array(curr) - np.array(prev)
            if np.sum(np.abs(deltas)) != 1:
                raise OperatorException(
                    f'Sites {prev} and {curr} are not adjacent.')
            d = np.argmax(np.abs(deltas))
            edge_list.append((*prev, d) if deltas[d] > 0 else (*curr, d))
            prev = curr
        self._apply_edge_operators_edges(edge_list)
    

    def apply_z1z2(self, s0, s1):
        '''Apply a Z1Z2 operation (correlated syndromes)'''
        key = (s1, s0) if s0[0] > s1[0] else (s0, s1)
        self.correlated_syndromes[key] = not self.correlated_syndromes[key]
    

    def measure_syndrome(self, failed_sites=None):
        '''
        Measure the error syndrome
        
        Takes into account correlated errors and possibility of measurement failure
        Repeats syndrome measurements until success is achieved at all positions
        
        Keyword arguments:
        failed -- coordinates/mask for syndrome measurements that will fail
        '''
        syndrome = super().measure_syndrome()
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
            self.measure_syndrome_again(x, y, tries=tries+1)
        return tries


    def apply_errors(self):
        '''Apply errors on edge operators according to self.p and self.correlated_syndrome_p'''
        super().apply_errors()
        for key, p in self.correlated_syndrome_p.items():
            if p > np.random.rand():
                self.apply_z1z2(*key)


    def reset(self):
        '''Reset the error state of the lattice'''
        super().reset()
        self.correlated_syndrome_p = defaultdict(float)
        self.correlated_syndromes = defaultdict(bool)


class PhotonicPlanarCode(PlanarCode):

    def __init__(self, L, W=None, boundaries=DEFAULT_BOUNDS, pz=0.05, px=None, rz=0, rx=None):
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
        p0 = 1 - (1-r)**5
        
        if dual.is_real_edge(z[0]):
            CorrelatedPlanarLattice.dump_error(dual.p, z[0], r*(1-r) / (2*p0))
        
        if dual.is_real_edge(z[3]):
            CorrelatedPlanarLattice.dump_error(dual.p, z[3], r*(1-r)**3 / (2*p0))

        s0 = z[0][:2]
        s1 = z[1][:2]
        if dual.is_real_site(s0) and dual.is_real_site(s1):
            CorrelatedPlanarLattice.dump_error(
                dual.correlated_syndrome_p, (s0, s1), r*(1-r)**2 / (2*p0))

    
    def measure_syndrome(self, **kwargs):
        '''Measure the error syndrome at the current time.
        Also propagates errors probabilities from failed measurements.
        
        Nontrivial syndromes on the primal and dual lattices are represented by True values.

        Returns arrays of error syndromes for both primal and dual lattices
        '''
        syndromes = []
        for i, (syndrome, retries) in enumerate(super().measure_syndrome(**kwargs)):
            print(retries)
            for (x, y), tries in retries.items():
                for _ in range(tries):
                    self.apply_failed_measurement_error(x, y, reverse=bool(i))
            syndromes.append(syndrome)
        return syndromes
