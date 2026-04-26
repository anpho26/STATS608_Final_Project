import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .radon_ops import circle_mask, radon_rows, backproject_single
from .shapes import random_init
from .metrics import aligned_mixture2_error, projection_domain_error_mixture2


def em_reconstruct_mixture2(
    Y,
    candidate_angles,
    output_size,
    true_classes=None,
    true_angles=None,
    true_img1=None,
    true_img2=None,
    n_em=100,
    n_inner=50,
    lr=1e-4,
    lam=5e-3,
    temp_start=2.0,
    temp_end=0.2,
    temp_decay=0.995,
    sigma2=None,
    seed=0,
    verbose=False,
    verbose_tqdm=False,
    x1_init=None,
    x2_init=None,
    pi_init=(0.5, 0.5),
    show_every=10,
    align_angle_grid=None,
):
    n_obs, _ = Y.shape
    n_angles = len(candidate_angles)
    mask = circle_mask(output_size)

    x1 = random_init(output_size, seed=seed) if x1_init is None else x1_init.copy()
    x2 = random_init(output_size, seed=seed + 1) if x2_init is None else x2_init.copy()

    x1 = np.where(mask, x1, 0.0)
    x2 = np.where(mask, x2, 0.0)

    if sigma2 is None:
        sigma2 = np.var(Y)

    pi = np.array(pi_init, dtype=float)
    pi /= pi.sum()

    obj_hist = []
    proj_err_hist = []
    align_err_hist = []
    align_match_hist = []

    x1_hist = [x1.copy()]
    x2_hist = [x2.copy()]

    iterator = tqdm(range(n_em), desc="EM mixture") if verbose_tqdm else range(n_em)

    for em_it in iterator:
        temperature = max(temp_end, temp_start * (temp_decay ** em_it))

        pred1 = radon_rows(x1, candidate_angles)
        pred2 = radon_rows(x2, candidate_angles)

        d2_1 = np.sum((Y[:, None, :] - pred1[None, :, :]) ** 2, axis=2)
        d2_2 = np.sum((Y[:, None, :] - pred2[None, :, :]) ** 2, axis=2)

        log_r1 = np.log(pi[0] + 1e-16) - 0.5 * d2_1 / (temperature * sigma2)
        log_r2 = np.log(pi[1] + 1e-16) - 0.5 * d2_2 / (temperature * sigma2)

        both = np.concatenate([log_r1, log_r2], axis=1)
        both -= both.max(axis=1, keepdims=True)
        both = np.exp(both)
        both /= both.sum(axis=1, keepdims=True)

        R1 = both[:, :n_angles]
        R2 = both[:, n_angles:]

        pi[0] = R1.sum() / n_obs
        pi[1] = R2.sum() / n_obs
        pi /= pi.sum()

        B1 = R1.T @ Y
        B2 = R2.T @ Y
        counts1 = R1.sum(axis=0)
        counts2 = R2.sum(axis=0)

        for _ in range(n_inner):
            pred1 = radon_rows(x1, candidate_angles)
            grad1 = np.zeros_like(x1)

            for m, ang in enumerate(candidate_angles):
                if counts1[m] < 1e-12:
                    continue

                resid1 = counts1[m] * pred1[m] - B1[m]
                grad1 += backproject_single(resid1, ang, output_size)

            grad1 = grad1 / sigma2 + lam * x1
            x1 = x1 - lr * grad1
            x1 = np.clip(x1, 0.0, None)
            x1 = np.where(mask, x1, 0.0)

        for _ in range(n_inner):
            pred2 = radon_rows(x2, candidate_angles)
            grad2 = np.zeros_like(x2)

            for m, ang in enumerate(candidate_angles):
                if counts2[m] < 1e-12:
                    continue

                resid2 = counts2[m] * pred2[m] - B2[m]
                grad2 += backproject_single(resid2, ang, output_size)

            grad2 = grad2 / sigma2 + lam * x2
            x2 = x2 - lr * grad2
            x2 = np.clip(x2, 0.0, None)
            x2 = np.where(mask, x2, 0.0)

        pred1 = radon_rows(x1, candidate_angles)
        pred2 = radon_rows(x2, candidate_angles)

        d2_1 = np.sum((Y[:, None, :] - pred1[None, :, :]) ** 2, axis=2)
        d2_2 = np.sum((Y[:, None, :] - pred2[None, :, :]) ** 2, axis=2)

        obj = (
            0.5 * np.sum(R1 * d2_1) / sigma2
            + 0.5 * np.sum(R2 * d2_2) / sigma2
            + 0.5 * lam * (np.sum(x1**2) + np.sum(x2**2))
        )
        obj_hist.append(obj)

        x1_hist.append(x1.copy())
        x2_hist.append(x2.copy())

        cls_est = np.where(R1.sum(axis=1) >= R2.sum(axis=1), 1, 2)
        ang_est_idx = np.argmax(np.concatenate([R1, R2], axis=1), axis=1) % n_angles
        ang_est = candidate_angles[ang_est_idx]

        proj_err = projection_domain_error_mixture2(Y, x1, x2, cls_est, ang_est)
        proj_err_hist.append(proj_err)

        if true_img1 is not None and true_img2 is not None:
            align_out = aligned_mixture2_error(
                x1, x2, true_img1, true_img2, angle_grid=align_angle_grid
            )
            align_err_hist.append(align_out["total_err"])
            align_match_hist.append(align_out["match"])
        else:
            align_out = None
            align_err_hist.append(np.nan)
            align_match_hist.append(None)

        if verbose_tqdm:
            postfix = {
                "temp": f"{temperature:.3f}",
                "obj": f"{obj:.2f}",
                "proj": f"{proj_err:.4f}",
                "pi1": f"{pi[0]:.2f}",
                "pi2": f"{pi[1]:.2f}",
            }

            if align_out is not None:
                postfix["align"] = f"{align_out['total_err']:.4f}"

            iterator.set_postfix(postfix)

        if verbose and (((em_it + 1) % show_every == 0) or (em_it == 0)):
            msg = (
                f"EM iter {em_it+1:04d} | "
                f"temp={temperature:.4f} | "
                f"obj={obj:.4f} | "
                f"proj_err={proj_err:.6f} | "
                f"pi=({pi[0]:.3f},{pi[1]:.3f})"
            )

            if align_out is not None:
                msg += f" | align_err={align_out['total_err']:.6f} | match={align_out['match']}"

            print(msg)

    return {
        "x1_hist": x1_hist,
        "x2_hist": x2_hist,
        "x1": x1,
        "x2": x2,
        "R1": R1,
        "R2": R2,
        "pi": pi,
        "obj_hist": obj_hist,
        "proj_err_hist": proj_err_hist,
        "align_err_hist": align_err_hist,
        "align_match_hist": align_match_hist,
        "class_est": cls_est,
        "angle_est": ang_est,
    }