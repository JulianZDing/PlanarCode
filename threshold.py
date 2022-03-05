import os
import pickle
import multiprocessing as mp

from collections import defaultdict
from scipy.stats import linregress
from tqdm.auto import tqdm

from photonic_planar_code import PhotonicPlanarCode
from graph_operations import *
from visualization import *


def single_shot_correction(code, **kwargs):
    '''
    Perform a single error-correction cycle on the code

    Arguments:
    code -- instance of PhotonicPlanarCode

    Returns if the error correction procedure resulted in a logical error
    '''
    code.reset()
    code.advance(1)
    code.measure_syndrome(lattice=1)
    lattice = code.primal
    syndrome = lattice.measure_syndrome(debug=True)
    matching, paths = min_weight_syndrome_matching(lattice, syndrome, **kwargs)
    for path in paths:
        lattice.apply_edge_operators(path)
    lattice.reset_error_model()
    return lattice.detect_logical_errors()


def sample_logical_errors(L, p, r, samples=1000, **kwargs):
    '''
    Generate a planar code and perform error correction, sampling for logical error rate

    Arguments:
    L -- side length of lattice
    p -- physical (memory) error probability
    r -- probability photon is lost in a waveguide
    samples -- number of times to sample
    '''
    code = PhotonicPlanarCode(L, pz=p, rz=r)
    logical_errors = [single_shot_correction(code, **kwargs) for _ in range(samples)]
    return np.array(logical_errors)


def threshold_slope(p, r, lengths=[4,6,8], **kwargs):
    x = lengths
    y = [np.mean(sample_logical_errors(L, p, r, **kwargs)) for L in lengths]
    regress = linregress(x, y)
    return regress.slope


def above_threshold(p, r, small_L=4, large_L=8, samples=1000, **kwargs):
    small_lattice_errors = sample_logical_errors(small_L, p, r, samples, **kwargs)
    large_lattice_errors = sample_logical_errors(large_L, p, r, samples, **kwargs)
    is_above_threshold = np.mean(large_lattice_errors) > np.mean(small_lattice_errors)
    return is_above_threshold


def boundary_search(
    p_range, r_range, threshold,
    split=2, max_depth=8, result_key=None, cache_dir=None, save_every=0, verbose=True, **kwargs
):
    '''
    Recursively resolve the phase-transition boundary within a search space of p and r values
    Arguments:
    p_range -- (min, max) tuple for the search range over p
    r_range -- (min, max) tuple for the search range over r
    threshold -- function handle for a function f(p, r): bool
                 that determines if (p, r) is below or above threshold
    
    Keyword arguments:
    split -- number of partitions to make in each dimension
    max_depth -- maximum number of splits
    result_key -- function handle for extracting a boolean value for above/below threshold
                  from the result of the threshold function
    cache_dir -- where to store and load cache data
    verbose -- whether or not to print progress bars
    '''
    def gen_filename(split_char='_'):
        p0, p1 = p_range
        r0, r1 = r_range
        parts = [
            threshold.__name__, 'boundary_search',
            p0, p1, r0, r1, split, max_depth
        ]
        return split_char.join([str(part) for part in parts]) + '.pkl'
            
    def generate_ids(offset, depth):
        step = split ** (max_depth - depth)
        ids = [offset]
        for i in range(1, split+1):
            ids.append(offset + i*step)
        return ids

    RESULTS = 'results'
    DEPTH = 'depth'
    WORKLIST = 'worklist'

    def save_cache(results, depth, worklist, cache_dir):
        if cache_dir is None:
            print('No cache directory provided, cannot save progress.')
        else:
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            cache_path = os.path.join(cache_dir, gen_filename())
            print(f'Caching progress to {cache_path}.')
            with open(cache_path, 'wb') as cache:
                to_cache = {RESULTS: results, DEPTH: depth, WORKLIST: {**worklist}}
                pickle.dump(to_cache, cache)

    def load_cache(cache_dir):
        cache_file = gen_filename()
        if cache_dir is None or not os.path.isfile(os.path.join(cache_dir, cache_file)):
            print(
                ('No cache directory provided'
                if cache_dir is None else
                f'Cache file {os.path.join(cache_dir, cache_file)} not found')
                + ', starting from scratch.'
            )
            results = {}
            depth = 1
            starting_ids = generate_ids(0, 1)
            worklist = defaultdict(list, {1: [(starting_ids, starting_ids)]})
        else:
            cache_path = os.path.join(cache_dir, cache_file)
            print(f'Loading cached progress from {cache_path}.')
            with open(cache_path, 'rb') as cache:
                data = pickle.load(cache)
                results = data[RESULTS]
                depth = data[DEPTH]
                worklist = defaultdict(list, data[WORKLIST])
        return results, depth, worklist


    grid_size = split ** max_depth + 1
    P = np.linspace(*p_range, grid_size)
    R = np.linspace(*r_range, grid_size)
    results, depth, worklist = load_cache(cache_dir)
    counter = 0
    try:
        while depth <= max_depth:
            next_depth = depth+1
            this_worklist = tqdm(worklist[depth], f'Depth {depth}') if verbose else worklist[depth]
            for p_ids, r_ids in this_worklist:
                combos = [(i, j) for i in p_ids for j in r_ids]
                if verbose:
                    combos = tqdm(combos, desc=f'Depth {depth} sub-progress')
                for i, j in combos:
                    key = (i, j)
                    if key not in results:
                        results[key] = threshold(P[i], R[j], **kwargs)
                        counter += 1
                    if save_every > 0 and counter >= save_every:
                        save_cache(results, depth, worklist, cache_dir)
                        counter = 0

                if depth != max_depth:
                    for p_idx, i in enumerate(p_ids[:-1]):
                        for r_idx, j in enumerate(r_ids[:-1]):
                            next_i = p_ids[p_idx+1]
                            next_j = r_ids[r_idx+1]
                            corners = [
                                results[(i,j)], results[(next_i,j)],
                                results[(i,next_j)], results[(next_i,next_j)]
                            ]
                            if result_key:
                                corners = [result_key(el) for el in corners]
                            if any([corners[0] != t for t in corners[1:]]):
                                worklist[next_depth].append(
                                    (generate_ids(i, next_depth), generate_ids(j, next_depth)))
            depth = next_depth
    except KeyboardInterrupt:
        print(f'Interrupted at depth {depth}')
    finally:
        print('Attempting to cache progress...')
        save_cache(results, depth, worklist, cache_dir)

    return {(P[i], R[j]): thresh for (i, j), thresh in results.items()}


class BoundarySearchEngine():

    RESULTS = 'results'
    DEPTH = 'depth'
    WORKLIST = 'worklist'

    def __init__(self, p_range, r_range, split=2, max_depth=8, cache_dir=None, workers=1):
        '''
        Initialize boundary search engine

        Arguments:
        p_range -- (min, max) tuple for the search range over p
        r_range -- (min, max) tuple for the search range over r

        Keyword arguments:
        split -- number of partitions to make in each dimension
        max_depth -- maximum number of splits
        cache_dir -- where to store and load cache data
        workers -- number of workers to use in multiprocessing
        '''
        self.p_range = p_range
        self.r_range = r_range
        self.split = split
        self.max_depth = max_depth

        grid_size = split ** max_depth + 1
        self.P = np.linspace(*p_range, grid_size)
        self.R = np.linspace(*r_range, grid_size)

        self.cache_dir = cache_dir
        self.workers = workers


    def _gen_filename(self, threshold, split_char='_'):
        p0, p1 = self.p_range
        r0, r1 = self.r_range
        parts = [
            threshold.__name__, 'boundary_search',
            p0, p1, r0, r1, self.split, self.max_depth
        ]
        return split_char.join([str(part) for part in parts]) + '.pkl'


    def _generate_ids(self, offset, depth):
        step = self.split ** (self.max_depth - depth)
        ids = [offset]
        for i in range(1, self.split+1):
            ids.append(offset + i*step)
        return ids
    

    def _save_cache(self, results, depth, worklist, threshold):
        if self.cache_dir is None:
            print('No cache directory provided, cannot save progress.')
        else:
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, self._gen_filename(threshold))
            print(f'Caching progress to {cache_path}.')
            with open(cache_path, 'wb') as cache:
                to_cache = {
                    self.RESULTS: results, self.DEPTH: depth, self.WORKLIST: dict(worklist)}
                pickle.dump(to_cache, cache)


    def _load_cache(self, threshold):
        cache_file = self._gen_filename(threshold)
        if self.cache_dir is None or not os.path.isfile(os.path.join(self.cache_dir, cache_file)):
            print(
                ('No cache directory provided'
                if self.cache_dir is None else
                f'Cache file {os.path.join(self.cache_dir, cache_file)} not found')
                + ', starting from scratch.'
            )
            results = {}
            depth = 1
            starting_ids = self._generate_ids(0, 1)
            worklist = defaultdict(list, {1: [(starting_ids, starting_ids)]})
        else:
            cache_path = os.path.join(self.cache_dir, cache_file)
            print(f'Loading cached progress from {cache_path}.')
            with open(cache_path, 'rb') as cache:
                data = pickle.load(cache)
                results = data[self.RESULTS]
                depth = data[self.DEPTH]
                worklist = data[self.WORKLIST]
        worklist = defaultdict(list, worklist)
        return results, depth, worklist


    def _compute_point(self, key, results, threshold, kwargs):
        i, j = key
        thresh = threshold(self.P[i], self.R[j], **kwargs) if key not in results else None
        return key, thresh


    def boundary_search(self, threshold, result_key=None, save_every=0, verbose=True, **kwargs):
        '''
        Recursively resolve the phase-transition boundary within a search space of p and r values

        Arguments:
        threshold -- function handle for a function f(p, r): bool
                    that determines if (p, r) is below or above threshold
        
        Keyword arguments:
        result_key -- function handle for extracting a boolean value for above/below threshold
                    from the result of the threshold function
        save_every -- save frequency (default: save only at the end)
        verbose -- whether or not to print progress bars
        '''
        results, depth, worklist = self._load_cache(threshold)
        pool = mp.Pool(min(self.workers, mp.cpu_count()))
        counter = 0
        try:
            while depth <= self.max_depth:
                next_depth = depth+1
                this_worklist = tqdm(worklist[depth], f'Depth {depth}') if verbose else worklist[depth]
                for p_ids, r_ids in this_worklist:
                    combos = [(i, j) for i in p_ids for j in r_ids]
                    arg_list = [(key, results, threshold, kwargs) for key in combos]
                    thresholds = pool.starmap(self._compute_point, arg_list)
                    for key, thresh in thresholds:
                        if thresh is not None:
                            results[key] = thresh
                            counter += 1
                    if save_every > 0 and counter >= save_every:
                        self._save_cache(results, depth, worklist, threshold)
                        counter = 0
                    if depth != self.max_depth:
                        for p_idx, i in enumerate(p_ids[:-1]):
                            for r_idx, j in enumerate(r_ids[:-1]):
                                next_i = p_ids[p_idx+1]
                                next_j = r_ids[r_idx+1]
                                corners = [
                                    results[(i,j)], results[(next_i,j)],
                                    results[(i,next_j)], results[(next_i,next_j)]
                                ]
                                if result_key:
                                    corners = [result_key(el) for el in corners]
                                if any([corners[0] != t for t in corners[1:]]):
                                    worklist[next_depth].append(
                                        (self._generate_ids(i, next_depth), self._generate_ids(j, next_depth)))
                depth = next_depth
        except KeyboardInterrupt:
            print(f'Interrupted at depth {depth}')
        finally:
            self._save_cache(results, depth, worklist, threshold)

        return {(self.P[i], self.R[j]): thresh for (i, j), thresh in results.items()}
