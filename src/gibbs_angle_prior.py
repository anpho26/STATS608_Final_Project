import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from tqdm.auto import tqdm

from src.utils import circle_mask, radon_rows, random_init, backproject_single
from src.gibbs import (
    pack_image,
    radon_matrix,
    gibbs_sample_radon_known_angles,
    mala_step,
)


# ------------------------------------------------------------
# Helper: DP-uniform-base posterior update on finite angle grid
# ------------------------------------------------------------

def sample_angle_weights_dp_uniform(counts, alpha_angle):
    """
    Finite-grid approximation to

        G ~ DP(alpha_angle, Uniform)

    On K candidate angles, this becomes

        pi_angle ~ Dirichlet(alpha_angle / K, ..., alpha_angle / K)

    Posterior update:

        pi_angle | z ~ Dirichlet(alpha_angle / K + counts)
    """
    k = len(counts)
    base_weights = np.ones(k) / k
    return np.random.dirichlet(alpha_angle * base_weights + counts)


# ------------------------------------------------------------
# Vanilla Gibbs sampler with DP-uniform-base angle prior
# ------------------------------------------------------------

def vanilla_gibbs_sampler_angle_prior(
    data,
    candidate_angles,
    sigma2,
    sigma_eps2,
    n_samples=1,
    n_burnins=0,
    random_state=None,
    x_init=None,
    alpha_angle=100.0,
    verbose=0,
    imshow=True,
    imsave=False,
    dir="plotsVG_angle_prior",
):
    np.random.seed(random_state)

    n, d = data.shape
    k = len(candidate_angles)

    mask_d = circle_mask(d)
    p = mask_d.sum()

    radon_map = radon_matrix(d, candidate_angles)
    os.makedirs(dir, exist_ok=True)

    x_sum = np.zeros((d, d))

    x = np.random.standard_normal(p) * np.sqrt(sigma2) if x_init is None else x_init.copy()
    if len(x.shape) == 2:
        x = pack_image(x, d)

    # z stores angle indices, not angle values
    z = np.random.randint(k, size=n)

    # Initial angle distribution centered at uniform
    pi_angle = np.ones(k) / k

    samples_x = []
    samples_z = []
    samples_pi_angle = []

    for it in tqdm(range(n_samples + n_burnins), desc="Vanilla Gibbs + DP angle prior"):

        # 1. Sample z | x, y, pi_angle
        sino = (x @ radon_map.T).reshape((k, d))

        for i in range(n):
            resid = data[i] - sino

            logp = -0.5 / sigma_eps2 * np.sum(resid * resid, axis=1)
            logp += np.log(pi_angle + 1e-300)
            logp -= logsumexp(logp)

            z[i] = np.random.choice(k, p=np.exp(logp))

        counts = np.bincount(z, minlength=k)

        # 2. Sample pi_angle | z under DP(alpha, Uniform)
        pi_angle = sample_angle_weights_dp_uniform(counts, alpha_angle)

        # 3. Sample x | z, y
        z_angles = candidate_angles[z]

        x = gibbs_sample_radon_known_angles(
            data,
            z_angles,
            candidate_angles,
            sigma2=sigma2,
            sigma_eps2=sigma_eps2,
            radon_map=radon_map,
        )[0]

        if it >= n_burnins:
            x_sum += x

        samples_x.append(x.copy())
        samples_z.append(z.copy())
        samples_pi_angle.append(pi_angle.copy())

        x = pack_image(x, d)

        if verbose > 0 and it % verbose == 0:
            if it >= n_burnins:
                fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.7))

                axes[0].imshow(samples_x[-1], cmap="gray")
                axes[0].set_title("Current sample")
                axes[0].axis("off")

                axes[1].imshow(x_sum / (it + 1 - n_burnins), cmap="gray")
                axes[1].set_title("Running mean")
                axes[1].axis("off")

                axes[2].bar(np.arange(k), counts, width=1.0, linewidth=0)
                axes[2].set_title("Label count")
                axes[2].set_xticks([])

                axes[3].bar(np.arange(k), pi_angle, width=1.0, linewidth=0)
                axes[3].set_title("DP angle weights")
                axes[3].set_xticks([])

                plt.suptitle(f"Vanilla Gibbs + DP angle prior iter {it}")
                plt.tight_layout()

                if imsave:
                    plt.savefig(f"{dir}/VG_DP_angle_prior_iter{it}.png")

                if imshow:
                    plt.show()

                plt.close()

    return (
        samples_x[n_burnins:],
        samples_z[n_burnins:],
        samples_pi_angle[n_burnins:],
    )


# ------------------------------------------------------------
# Gibbs-LD sampler with DP-uniform-base angle prior
# ------------------------------------------------------------

def gibbs_LD_sampler_angle_prior(
    data,
    candidate_angles,
    n_gibbs=100,
    n_burnins=0,
    n_inner=50,
    lr=1e-4,
    lam=5e-3,
    temp_start=2.0,
    temp_end=1.0,
    temp_decay=0.995,
    seed=0,
    sigma2=None,
    x_init=None,
    alpha_angle=100.0,
    verbose=-1,
    imshow=True,
    imsave=False,
    dir="plotsGLD_angle_prior",
):
    np.random.seed(seed)

    n, d = data.shape
    k = len(candidate_angles)

    mask = circle_mask(d)
    plot_x = list(range(k))

    if sigma2 is None:
        sigma2 = np.var(data)

    os.makedirs(dir, exist_ok=True)

    x = random_init(d, seed=seed) if x_init is None else x_init.copy()
    x = np.where(mask, x, 0.0)

    # z stores angle indices
    z = np.random.randint(k, size=n)

    # Initial angle distribution centered at uniform
    pi_angle = np.ones(k) / k

    xs = [x.copy()]
    zs = [z.copy()]
    pis_angle = [pi_angle.copy()]
    x_mean = np.zeros_like(x)

    pbar = tqdm(range(n_gibbs + n_burnins), desc="Gibbs LD + DP angle prior")

    for it in pbar:
        temperature = max(temp_end, temp_start * (temp_decay ** it))

        # 1. Sample z | x, y, pi_angle
        pred = radon_rows(x, candidate_angles)

        for i in range(n):
            resid = data[i] - pred

            logp = -0.5 * np.sum(resid * resid, axis=1) / (temperature * sigma2)
            logp += np.log(pi_angle + 1e-300)
            logp -= logsumexp(logp)

            z[i] = np.random.choice(np.arange(k), p=np.exp(logp))

        counts = np.bincount(z, minlength=k)

        # 2. Sample pi_angle | z under DP(alpha, Uniform)
        pi_angle = sample_angle_weights_dp_uniform(counts, alpha_angle)

        # 3. Update x | z, y using Langevin dynamics
        R = np.zeros((n, k), dtype=int)
        R[np.arange(n), z] = 1

        B = R.T @ data

        for _ in range(n_inner):
            pred = radon_rows(x, candidate_angles)
            grad = np.zeros_like(x)

            for m, ang in enumerate(candidate_angles):
                if counts[m] < 1e-12:
                    continue

                resid = counts[m] * pred[m] - B[m]
                grad += backproject_single(resid, ang, d)

            grad = grad / sigma2 + lam * x

            x = x - lr * grad + np.sqrt(2 * lr) * np.random.randn(*x.shape)
            x = np.where(mask, x, 0.0)

        xs.append(x.copy())
        zs.append(z.copy())
        pis_angle.append(pi_angle.copy())

        if it >= n_burnins:
            x_mean += x.copy()

        pbar.set_postfix({
            "temp": f"{temperature:.3f}",
            "occupied": int(np.sum(counts > 0)),
        })

        if verbose >= 0:
            if ((it + 1) % verbose == 0) or (it == 0):
                if it >= n_burnins:
                    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.7))

                    axes[0].imshow(xs[-1], cmap="gray")
                    axes[0].set_title("Current sample")
                    axes[0].axis("off")

                    axes[1].imshow(x_mean / (it + 1 - n_burnins), cmap="gray")
                    axes[1].set_title("Running mean")
                    axes[1].axis("off")

                    axes[2].bar(plot_x, counts, width=1.0, linewidth=0)
                    axes[2].set_title("Label count")
                    axes[2].set_xticks([])

                    axes[3].bar(plot_x, pi_angle, width=1.0, linewidth=0)
                    axes[3].set_title("DP angle weights")
                    axes[3].set_xticks([])

                    plt.suptitle(f"Gibbs LD + DP angle prior iter {it}")
                    plt.tight_layout()

                    if imsave:
                        plt.savefig(f"{dir}/GLD_DP_angle_prior_iter{it}.png")

                    if imshow:
                        plt.show()

                    plt.close()

    return (
        xs[n_burnins + 1:],
        zs[n_burnins + 1:],
        pis_angle[n_burnins + 1:],
    )


# ------------------------------------------------------------
# MALA sampler with DP-uniform-base angle prior
# ------------------------------------------------------------

def MALA_angle_prior(
    data,
    candidate_angles,
    n_gibbs=100,
    n_burnins=0,
    n_inner=50,
    lr=1e-4,
    lam=5e-3,
    temp_start=2.0,
    temp_end=1.0,
    temp_decay=0.995,
    seed=0,
    sigma2=None,
    x_init=None,
    alpha_angle=100.0,
    verbose=-1,
    imshow=True,
    imsave=False,
    dir="plotsMALA_angle_prior",
):
    np.random.seed(seed)

    n, d = data.shape
    k = len(candidate_angles)

    mask = circle_mask(d)
    plot_x = list(range(k))

    if sigma2 is None:
        sigma2 = np.var(data)

    os.makedirs(dir, exist_ok=True)

    x = random_init(d, seed=seed) if x_init is None else x_init.copy()
    x = np.where(mask, x, 0.0)

    # z stores angle indices
    z = np.random.randint(k, size=n)

    # Initial angle distribution centered at uniform
    pi_angle = np.ones(k) / k

    xs = [x.copy()]
    zs = [z.copy()]
    pis_angle = [pi_angle.copy()]
    x_mean = np.zeros_like(x)

    pbar = tqdm(range(n_gibbs + n_burnins), desc="MALA + DP angle prior")

    for it in pbar:
        temperature = max(temp_end, temp_start * (temp_decay ** it))

        # 1. Sample z | x, y, pi_angle
        pred = radon_rows(x, candidate_angles)

        for i in range(n):
            resid = data[i] - pred

            logp = -0.5 * np.sum(resid * resid, axis=1) / (temperature * sigma2)
            logp += np.log(pi_angle + 1e-300)
            logp -= logsumexp(logp)

            z[i] = np.random.choice(np.arange(k), p=np.exp(logp))

        counts = np.bincount(z, minlength=k)

        # 2. Sample pi_angle | z under DP(alpha, Uniform)
        pi_angle = sample_angle_weights_dp_uniform(counts, alpha_angle)

        # 3. Update x | z, y using MALA
        accepts = 0

        for _ in range(n_inner):
            x, accepted = mala_step(
                x,
                data,
                z,
                candidate_angles,
                sigma2,
                lam,
                lr,
                mask,
            )
            accepts += accepted

        acc_rate = accepts / n_inner

        xs.append(x.copy())
        zs.append(z.copy())
        pis_angle.append(pi_angle.copy())

        if it >= n_burnins:
            x_mean += x.copy()

        pbar.set_postfix({
            "temp": f"{temperature:.3f}",
            "acc": f"{acc_rate:.2f}",
            "occupied": int(np.sum(counts > 0)),
        })

        if verbose >= 0:
            if ((it + 1) % verbose == 0) or (it == 0):
                if it >= n_burnins:
                    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.7))

                    axes[0].imshow(xs[-1], cmap="gray")
                    axes[0].set_title("Current sample")
                    axes[0].axis("off")

                    axes[1].imshow(x_mean / (it + 1 - n_burnins), cmap="gray")
                    axes[1].set_title("Running mean")
                    axes[1].axis("off")

                    axes[2].bar(plot_x, counts, width=1.0, linewidth=0)
                    axes[2].set_title("Label count")
                    axes[2].set_xticks([])

                    axes[3].bar(plot_x, pi_angle, width=1.0, linewidth=0)
                    axes[3].set_title("DP angle weights")
                    axes[3].set_xticks([])

                    plt.suptitle(f"MALA + DP angle prior iter {it}")
                    plt.tight_layout()

                    if imsave:
                        plt.savefig(f"{dir}/MALA_DP_angle_prior_iter{it}.png")

                    if imshow:
                        plt.show()

                    plt.close()

    return (
        xs[n_burnins + 1:],
        zs[n_burnins + 1:],
        pis_angle[n_burnins + 1:],
    )