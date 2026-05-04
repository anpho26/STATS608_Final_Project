import numpy as np
from scipy.ndimage import rotate
from src.utils import radon_rows

# Main case of 1 single true images
def measure_projection_error(x, z, data):
    pred = radon_rows(x, z)
    return np.sqrt(np.mean((data - pred)**2))

def measure_projection_error_minimal(x, data, candidate_angles):
    projs = radon_rows(x, candidate_angles)
    dists = np.mean((data[:, None, :]-projs[None, :, :])**2, axis=2)
    dists = dists.min(axis=1)
    return np.sqrt(np.mean(dists))

def measure_alignment_error(x, x_true, candidates_angles):
    best_err = np.inf
    for a in candidates_angles:
        xr = rotate(x, angle=a, reshape=False)
        err = np.linalg.norm(xr - x_true) / np.linalg.norm(x_true)
        best_err = min(best_err, err)
    return best_err

def measure_projection_error_batch(xs, zs, data):
    errs = []
    for x, z in zip(xs, zs):
        err = measure_projection_error(x, z, data)
        errs.append(err)
    return np.array(errs)

def measure_alignment_error_batch(xs, x_true, candidates_angles):
    errs = []
    for x in xs:
        err = measure_alignment_error(x, x_true, candidates_angles)
        errs.append(err)
    return np.array(errs)

# Extensions to mixtures of images
def measure_projection_error_multiclass(x, z, c, data):
    m = len(x)
    losses = []
    for i in range(m):
        taking = (c==i)
        losses.append(measure_projection_error(x[i], z[taking], data[taking])**2)
    return np.sqrt(np.mean(np.array(losses)))

def measure_projection_error_minimal_multiclass(x, data, candidate_angles): # Needs work
    projs = np.vstack([radon_rows(x_, candidate_angles) for x_ in x])
    dists = np.mean((data[:, None, :]-projs[None, :, :])**2, axis=2)
    dists = dists.min(axis=1)
    return np.sqrt(np.mean(dists))

def measure_alignment_error_multiclass(x, x_true, candidates_angles):
    return np.mean(np.array([measure_alignment_error(x_, x_true_, candidates_angles) for x_, x_true_ in zip(x, x_true)]))

def measure_projection_error_batch_multiclass(xs, zs, cs, data):
    errs = []
    for x, z, c in zip(xs, zs, cs):
        err = measure_projection_error_multiclass(x, z, z, data)
        errs.append(err)
    return np.array(errs)

def measure_alignment_error_batch_multiclass(xs, x_true, candidates_angles):
    errs = []
    for x in xs:
        err = measure_alignment_error_multiclass(x, x_true, candidates_angles)
        errs.append(err)
    return np.array(errs)