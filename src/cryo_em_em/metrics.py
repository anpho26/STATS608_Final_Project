import numpy as np
from scipy.ndimage import rotate
from .radon_ops import radon_rows


def aligned_relative_error(x, x_true, angle_grid=None):
    if angle_grid is None:
        angle_grid = np.arange(0, 180, 2)

    denom = np.linalg.norm(x_true)
    if denom == 0:
        return np.nan, np.nan

    best_err = np.inf
    best_angle = None

    for a in angle_grid:
        xr = rotate(x, angle=a, reshape=False, order=1)
        err = np.linalg.norm(xr - x_true) / denom

        if err < best_err:
            best_err = err
            best_angle = a

    return best_err, best_angle


def projection_domain_error(Y, x, est_angles):
    pred = radon_rows(x, est_angles)
    return np.mean((Y - pred) ** 2)


def aligned_mixture2_error(x1, x2, true_img1, true_img2, angle_grid=None):
    e11, a11 = aligned_relative_error(x1, true_img1, angle_grid)
    e22, a22 = aligned_relative_error(x2, true_img2, angle_grid)
    total_direct = e11 + e22

    e12, a12 = aligned_relative_error(x1, true_img2, angle_grid)
    e21, a21 = aligned_relative_error(x2, true_img1, angle_grid)
    total_swap = e12 + e21

    if total_direct <= total_swap:
        return {
            "total_err": total_direct,
            "match": "direct",
            "x1_err": e11,
            "x2_err": e22,
            "x1_angle": a11,
            "x2_angle": a22,
        }

    return {
        "total_err": total_swap,
        "match": "swapped",
        "x1_err": e12,
        "x2_err": e21,
        "x1_angle": a12,
        "x2_angle": a21,
    }


def projection_domain_error_mixture2(Y, x1, x2, cls_est, ang_est):
    pred = np.zeros_like(Y)

    idx1 = cls_est == 1
    idx2 = cls_est == 2

    if np.any(idx1):
        pred[idx1] = radon_rows(x1, ang_est[idx1])
    if np.any(idx2):
        pred[idx2] = radon_rows(x2, ang_est[idx2])

    return np.mean((Y - pred) ** 2)