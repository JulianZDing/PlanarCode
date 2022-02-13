from collections import defaultdict
from scipy.stats import ttest_ind

from photonic_planar_code import PhotonicPlanarCode
from graph_operations import *
from visualization import *

def single_shot_correction(code, z_state=0, **kwargs):
    '''
    Perform a single error-correction cycle on the code

    Arguments:
    code -- instance of PhotonicPlanarCode
    
    Keyword arguments:
    z_state -- whether or not there is already a logical Z encoded on the (primal) lattice
    pathfinding -- force a specific pathfinding algorithm (manhattan vs dijkstra)
    '''
    code.advance(1)
    for i in [0, 1]:
        syndrome = code.measure_syndrome(lattice=i)
        lattice = code.lattices[i]
        matching, paths = min_weight_syndrome_matching(lattice, syndrome, **kwargs)
        for path in paths:
            lattice.apply_edge_operators(path)
        lattice.reset_error_model()
    return lattice.detect_logical_errors(initial_z=z_state)


def sample_logical_errors(L, p, r, samples, track_z=False, **kwargs):
    '''
    Generate a planar code and perform error correction, sampling for logical error rate

    Arguments:
    L -- side length of lattice
    p -- physical (memory) error probability
    r -- probability photon is lost in a waveguide
    samples -- number of times to sample

    Keyword arguments:
    track_z -- Whether or not to persist logical errors between samples
               (code is reset between samples by default)
    '''
    code = PhotonicPlanarCode(L, pz=p, rz=r)
    logical_errors = []
    z_state = 0
    for _ in range(samples):
        z_error = single_shot_correction(code, z_state, **kwargs)
        if track_z and z_error:
            z_state = int(not bool(z_state))
        if not track_z:
            code.reset()
        logical_errors.append(z_error)
    return np.array(logical_errors)


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


def boundary_search(p_range, r_range, threshold, split=2, max_depth=8, **kwargs):
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
    '''
    grid_size = split ** max_depth + 1
    points = split + 1
    canvas = np.zeros((grid_size, grid_size))
    P = np.linspace(*p_range, grid_size)
    R = np.linspace(*r_range, grid_size)

    worklist = defaultdict(list, {1: [(p_range, r_range)]})
    results = {}
    for depth in range(1, max_depth+1):
        for pr, rr in worklist[depth]:
            p_array = np.linspace(*pr, points)
            r_array = np.linspace(*rr, points)

            box = np.zeros((points, points), dtype=bool)
            for i, p in enumerate(p_array):
                for j, r in enumerate(r_array):
                    key = (p, r)
                    if key in results:
                        thresh = results[key]
                    else:
                        thresh = threshold(p, r, **kwargs)
                        results[key] = thresh
                        p0 = p_array[i-1]
                        p1 = p_array[i]
                        r0 = r_array[j-1]
                        r1 = r_array[j]
                        select_p = (P >= p0) & (P < p1)
                        select_r = (R >= r0) & (R < r1)
                        sp, sr = np.meshgrid(select_p, select_r)
                        np.putmask(canvas, sp & sr, thresh)
                    box[i,j] = thresh

            if depth != max_depth:
                for i in range(points-1):
                    for j in range(points-1):
                        if any([box[i,j] != t for t in (box[i+1,j], box[i,j+1], box[i+1,j+1])]):
                            next_pr = (p_array[i], p_array[i+1])
                            next_rr = (r_array[j], r_array[j+1])
                            worklist[depth+1].append((next_pr, next_rr))
    return canvas, results
