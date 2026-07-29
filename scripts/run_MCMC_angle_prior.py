# In root dir, run:
# python3 -m scripts.run_MCMC_angle_prior

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from src.utils import (
    circle_mask,
    simulate_data,
    generate_4shape,
    generate_Sshape,
    load_image,
)

from src.gibbs_angle_prior import (
    vanilla_gibbs_sampler_angle_prior,
    gibbs_LD_sampler_angle_prior,
    MALA_angle_prior,
)

from src.cryo_em_em.em_single import em_reconstruct_live


# --------------------------------------------------
# Settings
# --------------------------------------------------

d = 32
n = 1000

seed_data = 345
seed_em = 100

sigma2 = 0.5
sigma_eps2 = sigma2

# Finite angle grid
candidate_angles = np.arange(0, 360, 4, dtype=float)

# MCMC seeds
seeds = [100]

# DP concentration parameter.
# Larger = closer to uniform.
# Smaller = more spiky / more preferred orientations.
alpha_angle = 100.0

Path("exp_output").mkdir(exist_ok=True)


# --------------------------------------------------
# Generate image
# --------------------------------------------------

# Option 1: generated non-symmetric image
img = generate_4shape(size=d)

# Option 2: S shape
# img = generate_Sshape(size=d)

# Option 3: load real image
# img = load_image("data/raw/13098l.png", size=d)

mask_d = circle_mask(d)
img = np.where(mask_d, img, 0.0)


plt.figure(figsize=(3, 3))
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("True image")
plt.tight_layout()
plt.savefig("exp_output/dp_angle_prior_true_image.png", dpi=150)
plt.close()


# --------------------------------------------------
# Simulate data with continuous uniform angles
# --------------------------------------------------

data, true_angles = simulate_data(
    img,
    n_obs=n,
    noise_std=np.sqrt(sigma2),
    seed=seed_data,
    angle_low=0.0,
    angle_high=360.0,
)

print("Data shape:", data.shape)
print("True angle range:", true_angles.min(), true_angles.max())

np.save("exp_output/single_data.npy", data)
np.save("exp_output/single_true_angles.npy", true_angles)


# --------------------------------------------------
# Run EM first for initialization
# --------------------------------------------------

em_dir = Path("exp_output/single_em_init_for_dp_angle_prior")
em_dir.mkdir(parents=True, exist_ok=True)

print("\nRunning EM initialization...")

xs_em, est_angles_em, R_em, metrics_em = em_reconstruct_live(
    data,
    candidate_angles,
    output_size=d,
    true_angles=true_angles,
    true_image=img,
    n_em=300,
    n_inner=50,
    lr=1e-4,
    lam=5e-3,
    temp_start=2.0,
    temp_end=0.2,
    temp_decay=0.99,
    seed=seed_em,
    verbose=False,
    verbose_tqdm=True,
)

x_init = xs_em[-1]

# Clean and normalize EM init
x_init = np.asarray(x_init, dtype=float)
x_init = np.where(mask_d, x_init, 0.0)

x_init -= x_init.min()
if x_init.max() > 0:
    x_init /= x_init.max()

np.save(em_dir / "x_em_final.npy", x_init)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("True image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(x_init, cmap="gray")
plt.title("EM initialization")
plt.axis("off")

plt.tight_layout()
plt.savefig(em_dir / "em_initialization.png", dpi=150)
plt.close()

print("EM initialization saved to:", em_dir / "x_em_final.npy")

if metrics_em is not None:
    if "obj_hist" in metrics_em:
        print("Final EM objective:", metrics_em["obj_hist"][-1])
    if "proj_err_hist" in metrics_em:
        print("Final EM projection error:", metrics_em["proj_err_hist"][-1])
    if "align_err_hist" in metrics_em:
        print("Final EM aligned error:", metrics_em["align_err_hist"][-1])


# --------------------------------------------------
# Helper plot functions
# --------------------------------------------------

def save_angle_prior_plot(pis_angle, out_dir, title):
    pis_angle = np.asarray(pis_angle)
    pi_mean = pis_angle.mean(axis=0)

    plt.figure(figsize=(8, 3))
    plt.bar(candidate_angles, pi_mean, width=4)
    plt.xlabel("Angle")
    plt.ylabel("Posterior mean weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/posterior_mean_dp_angle_weights.png", dpi=150)
    plt.close()


def save_final_reconstruction(xs, out_dir, title):
    xs = np.asarray(xs)

    x_last = xs[-1]
    x_mean = xs.mean(axis=0)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(x_last, cmap="gray")
    plt.title("Last sample")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(x_mean, cmap="gray")
    plt.title("Posterior mean")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/final_reconstruction_summary.png", dpi=150)
    plt.close()


# --------------------------------------------------
# Run Vanilla Gibbs with DP angle prior
# --------------------------------------------------

for seed in seeds:
    out_dir = (
        f"exp_output/single_gibbs_dp_angle_prior_"
        f"em_init_alpha{int(alpha_angle)}_seed{seed}"
    )

    print(f"\nRunning Vanilla Gibbs + DP angle prior, seed={seed}")

    xs_gibbs, zs_gibbs, pis_gibbs = vanilla_gibbs_sampler_angle_prior(
        data=data,
        candidate_angles=candidate_angles,
        sigma2=sigma2,
        sigma_eps2=sigma_eps2,
        n_samples=200,
        n_burnins=50,
        random_state=seed,
        x_init=x_init,
        alpha_angle=alpha_angle,
        verbose=20,
        imshow=False,
        imsave=True,
        dir=out_dir,
    )

    np.save(f"{out_dir}/xs_gibbs.npy", np.asarray(xs_gibbs))
    np.save(f"{out_dir}/zs_gibbs.npy", np.asarray(zs_gibbs))
    np.save(f"{out_dir}/pis_angle_gibbs.npy", np.asarray(pis_gibbs))

    save_angle_prior_plot(
        pis_gibbs,
        out_dir,
        "Vanilla Gibbs: posterior mean DP angle weights",
    )

    save_final_reconstruction(
        xs_gibbs,
        out_dir,
        "Vanilla Gibbs + DP angle prior",
    )

    print(f"Saved Vanilla Gibbs output to {out_dir}")


# --------------------------------------------------
# Run Gibbs-LD with DP angle prior
# --------------------------------------------------

for seed in seeds:
    out_dir = (
        f"exp_output/single_gibbs_ld_dp_angle_prior_"
        f"em_init_alpha{int(alpha_angle)}_seed{seed}"
    )

    print(f"\nRunning Gibbs-LD + DP angle prior, seed={seed}")

    xs_ld, zs_ld, pis_ld = gibbs_LD_sampler_angle_prior(
        data=data,
        candidate_angles=candidate_angles,
        n_gibbs=500,
        n_burnins=150,
        n_inner=50,
        lr=1e-6,
        lam=5e-3,
        temp_start=3.0,
        temp_end=1.0,
        temp_decay=0.99,
        seed=seed,
        sigma2=sigma2,
        x_init=x_init,
        alpha_angle=alpha_angle,
        verbose=50,
        imshow=False,
        imsave=True,
        dir=out_dir,
    )

    np.save(f"{out_dir}/xs_ld.npy", np.asarray(xs_ld))
    np.save(f"{out_dir}/zs_ld.npy", np.asarray(zs_ld))
    np.save(f"{out_dir}/pis_angle_ld.npy", np.asarray(pis_ld))

    save_angle_prior_plot(
        pis_ld,
        out_dir,
        "Gibbs-LD: posterior mean DP angle weights",
    )

    save_final_reconstruction(
        xs_ld,
        out_dir,
        "Gibbs-LD + DP angle prior",
    )

    print(f"Saved Gibbs-LD output to {out_dir}")


# --------------------------------------------------
# Run MALA with DP angle prior
# --------------------------------------------------

for seed in seeds:
    out_dir = (
        f"exp_output/single_mala_dp_angle_prior_"
        f"em_init_alpha{int(alpha_angle)}_seed{seed}"
    )

    print(f"\nRunning MALA + DP angle prior, seed={seed}")

    xs_mala, zs_mala, pis_mala = MALA_angle_prior(
        data=data,
        candidate_angles=candidate_angles,
        n_gibbs=500,
        n_burnins=150,
        n_inner=50,
        lr=1e-8,
        lam=5e-3,
        temp_start=3.0,
        temp_end=1.0,
        temp_decay=0.99,
        seed=seed,
        sigma2=sigma2,
        x_init=x_init,
        alpha_angle=alpha_angle,
        verbose=50,
        imshow=False,
        imsave=True,
        dir=out_dir,
    )

    np.save(f"{out_dir}/xs_mala.npy", np.asarray(xs_mala))
    np.save(f"{out_dir}/zs_mala.npy", np.asarray(zs_mala))
    np.save(f"{out_dir}/pis_angle_mala.npy", np.asarray(pis_mala))

    save_angle_prior_plot(
        pis_mala,
        out_dir,
        "MALA: posterior mean DP angle weights",
    )

    save_final_reconstruction(
        xs_mala,
        out_dir,
        "MALA + DP angle prior",
    )

    print(f"Saved MALA output to {out_dir}")