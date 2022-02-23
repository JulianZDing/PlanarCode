import os
import pickle

from collections import defaultdict
from scipy.stats import ttest_ind, linregress
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


def above_threshold(p, r, small_L=4, large_L=10, samples=1000, **kwargs):
    print(f'Threshold for {(p, r)} at {samples} samples', end='')
    small_lattice_errors = sample_logical_errors(small_L, p, r, samples, **kwargs)
    large_lattice_errors = sample_logical_errors(large_L, p, r, samples, **kwargs)
    is_above_threshold = (np.mean(small_lattice_errors) < np.mean(large_lattice_errors))
    _, p_val = ttest_ind(small_lattice_errors, large_lattice_errors)
    print(f' (t-test p-val: {p_val})')
    return is_above_threshold, p_val


def above_threshold_adaptive(
    p, r, small_L=4, large_L=10,
    min_samples=200, max_samples=int(1e5), multiplier=4,
    confidence=0.95, **kwargs
):
    '''
    Check if the combination of physical error probability and waveguide loss probability
    (p, r) is above threshold by comparing logical error rates between a small and large lattice.

    Arguments:
    p -- physical (memory) error probability
    r -- probability photon is lost in a waveguide

    Keyword arguments:
    small_L -- side length of small lattice
    large_L -- side length of large lattice
    min_samples -- number of samples to start trying with
    max_samples -- number of samples to stop trying
    multiplier -- multiplicative step for sample size between loops
    confidence -- target p value to stop sampling
    '''
    samples = min_samples
    p_val = 0

    while not np.isnan(p_val) and p_val < confidence:
        is_above_threshold, new_p_val = above_threshold(p, r, small_L, large_L, samples, **kwargs)
        if not np.isnan(new_p_val) and (new_p_val < p_val / 2 or samples == max_samples):
            break
        p_val = new_p_val
        samples = min(samples*multiplier, max_samples)

    return is_above_threshold, p_val


def boundary_search(
    p_range, r_range, threshold,
    split=2, max_depth=8, result_key=None, cache_dir=None, verbose=True, **kwargs
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
