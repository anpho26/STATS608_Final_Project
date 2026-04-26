import numpy as np
from tqdm.auto import tqdm

from .radon_ops import circle_mask, radon_rows, backproject_single
from .shapes import random_init
from .metrics import aligned_relative_error, projection_domain_error
from .plotting import show_em_status


def em_reconstruct_live(
    Y,
    candidate_angles,
    output_size,
    true_angles=None,
    true_image=None,
    n_em=100,
    n_inner=50,
    lr=1e-4,
    lam=5e-3,
    temp_start=2.0,
    temp_end=0.2,
    temp_decay=0.995,
    seed=0,
    show_every=10,
    sigma2=None,
    verbose=False,
    verbose_tqdm=False,
    x_init=None,
    align_angle_grid=None,
):
    n_obs, _ = Y.shape
    n_angles = len(candidate_angles)

    mask = circle_mask(output_size)

    x = random_init(output_size, seed=seed) if x_init is None else x_init.copy()
    x = np.where(mask, x, 0.0)

    if sigma2 is None:
        sigma2 = np.var(Y)

    R = np.full((n_obs, n_angles), 1.0 / n_angles)

    xs = [x.copy()]
    obj_hist = []
    proj_err_hist = []
    align_err_hist = []
    align_angle_hist = []

    iterator = tqdm(range(n_em), desc="EM iterations") if verbose_tqdm else range(n_em)

    for em_it in iterator:
        temperature = max(temp_end, temp_start * (temp_decay ** em_it))

        pred = radon_rows(x, candidate_angles)
        d2 = np.sum((Y[:, None, :] - pred[None, :, :]) ** 2, axis=2)

        log_resp = -0.5 * d2 / (temperature * sigma2)
        log_resp -= log_resp.max(axis=1, keepdims=True)

        R = np.exp(log_resp)
        R /= R.sum(axis=1, keepdims=True)

        B = R.T @ Y
        counts = R.sum(axis=0)

        for _ in range(n_inner):
            pred = radon_rows(x, candidate_angles)
            grad = np.zeros_like(x)

            for m, ang in enumerate(candidate_angles):
                if counts[m] < 1e-12:
                    continue

                resid = counts[m] * pred[m] - B[m]
                grad += backproject_single(resid, ang, output_size)

            grad = grad / sigma2 + lam * x
            x = x - lr * grad
            x = np.clip(x, 0.0, None)
            x = np.where(mask, x, 0.0)

        pred = radon_rows(x, candidate_angles)
        d2 = np.sum((Y[:, None, :] - pred[None, :, :]) ** 2, axis=2)

        obj = 0.5 * np.sum(R * d2) / sigma2 + 0.5 * lam * np.sum(x**2)
        obj_hist.append(obj)

        est_angles = candidate_angles[np.argmax(R, axis=1)]

        proj_err = projection_domain_error(Y, x, est_angles)
        proj_err_hist.append(proj_err)

        if true_image is not None:
            align_err, align_angle = aligned_relative_error(
                x, true_image, angle_grid=align_angle_grid
            )
        else:
            align_err, align_angle = np.nan, np.nan

        align_err_hist.append(align_err)
        align_angle_hist.append(align_angle)

        xs.append(x.copy())

        if verbose_tqdm:
            iterator.set_postfix({
                "temp": f"{temperature:.3f}",
                "obj": f"{obj:.2f}",
                "proj": f"{proj_err:.4f}",
                "align": f"{align_err:.4f}",
            })

        if verbose and (((em_it + 1) % show_every == 0) or (em_it == 0)):
            print(
                f"EM iter {em_it+1:04d} | "
                f"temp={temperature:.4f} | "
                f"obj={obj:.4f} | "
                f"proj_err={proj_err:.6f} | "
                f"align_err={align_err:.6f}"
            )
            show_em_status(
                iteration=em_it + 1,
                x=x,
                obj_hist=obj_hist,
                Y=Y,
                true_angles=true_angles if true_angles is not None else np.zeros(n_obs),
                est_angles=est_angles,
            )

    metrics = {
        "obj_hist": obj_hist,
        "proj_err_hist": proj_err_hist,
        "align_err_hist": align_err_hist,
        "align_angle_hist": align_angle_hist,
    }

    return xs, est_angles, R, metrics