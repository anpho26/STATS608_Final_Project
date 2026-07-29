# In root dir, run:
# python3 -m scripts.run_MCMC_nonuniform_angle_prior

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

# Full inference grid. The DP base measure is uniform on this grid.
candidate_angles = np.arange(0.0, 360.0, 4.0, dtype=float)

# Ground-truth truncated uniform angle distribution.
TRUE_ANGLE_LOW = 40.0
TRUE_ANGLE_HIGH = 120.0

# MCMC seeds.
seeds = [100, 200, 300]

# DP concentration parameter:
# larger values shrink more strongly toward the uniform base measure.
alpha_angle = 100.0

output_root = Path("exp_output/nonuniform_angle_prior")
output_root.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Generate ground-truth image
# --------------------------------------------------

img = generate_4shape(size=d)

# Alternatives:
# img = generate_Sshape(size=d)
# img = load_image("data/raw/13098l.png", size=d)

mask_d = circle_mask(d)
img = np.where(mask_d, img, 0.0)

plt.figure(figsize=(3, 3))
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("True image")
plt.tight_layout()
plt.savefig(output_root / "true_image.png", dpi=150)
plt.close()


# --------------------------------------------------
# Simulate data from a truncated uniform angle law
# --------------------------------------------------

data, true_angles = simulate_data(
    img,
    n_obs=n,
    noise_std=np.sqrt(sigma2),
    seed=seed_data,
    angle_low=TRUE_ANGLE_LOW,
    angle_high=TRUE_ANGLE_HIGH,
)

print("Data shape:", data.shape)
print(
    "Ground-truth angle interval:",
    f"[{TRUE_ANGLE_LOW:.1f}, {TRUE_ANGLE_HIGH:.1f})",
)
print(
    "Observed true-angle range:",
    f"[{true_angles.min():.3f}, {true_angles.max():.3f}]",
)

np.save(output_root / "data.npy", data)
np.save(output_root / "true_angles.npy", true_angles)

plt.figure(figsize=(8, 3))
plt.hist(
    true_angles,
    bins=np.arange(0.0, 364.0, 4.0),
    density=True,
    alpha=0.8,
)
plt.axvspan(
    TRUE_ANGLE_LOW,
    TRUE_ANGLE_HIGH,
    alpha=0.2,
    label="True support",
)
plt.axvline(TRUE_ANGLE_LOW, linestyle="--", linewidth=1)
plt.axvline(TRUE_ANGLE_HIGH, linestyle="--", linewidth=1)
plt.xlim(0.0, 360.0)
plt.xlabel("Angle (degrees)")
plt.ylabel("Density")
plt.title("Ground-truth truncated uniform angle distribution")
plt.legend()
plt.tight_layout()
plt.savefig(output_root / "true_angle_distribution.png", dpi=150)
plt.close()


# --------------------------------------------------
# Run EM first to obtain the image initialization
# --------------------------------------------------

em_dir = output_root / "em_init"
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
    show_every=10,
)

x_init = np.asarray(xs_em[-1], dtype=float)
x_init = np.where(mask_d, x_init, 0.0)

# Normalize only for initialization consistency.
x_init -= x_init.min()
if x_init.max() > 0:
    x_init /= x_init.max()

np.save(em_dir / "x_em_final.npy", x_init)
np.save(em_dir / "estimated_angles.npy", np.asarray(est_angles_em))
np.save(em_dir / "responsibilities.npy", np.asarray(R_em))

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

if metrics_em is not None:
    if "obj_hist" in metrics_em:
        print("Final EM objective:", metrics_em["obj_hist"][-1])
    if "proj_err_hist" in metrics_em:
        print("Final EM projection error:", metrics_em["proj_err_hist"][-1])
    if "align_err_hist" in metrics_em:
        print("Final EM aligned error:", metrics_em["align_err_hist"][-1])


# --------------------------------------------------
# Plotting helpers
# --------------------------------------------------

def discrete_true_angle_weights():
    """
    Discrete approximation of Uniform[TRUE_ANGLE_LOW, TRUE_ANGLE_HIGH)
    on the candidate-angle grid.
    """
    weights = np.zeros(len(candidate_angles), dtype=float)

    inside = (
        (candidate_angles >= TRUE_ANGLE_LOW)
        & (candidate_angles < TRUE_ANGLE_HIGH)
    )

    weights[inside] = 1.0

    if weights.sum() == 0:
        raise ValueError(
            "No candidate angles lie inside the true angle interval."
        )

    return weights / weights.sum()


def save_angle_distribution_comparison(
    pis_angle,
    zs,
    out_dir,
    title,
):
    """
    Compare:
      1. the true truncated-uniform grid weights,
      2. posterior mean DP weights,
      3. posterior mean sampled-label frequencies.
    """
    pis_angle = np.asarray(pis_angle)
    zs = np.asarray(zs)

    posterior_pi_mean = pis_angle.mean(axis=0)
    true_weights = discrete_true_angle_weights()

    label_counts = np.zeros(len(candidate_angles), dtype=float)
    for z_sample in zs:
        label_counts += np.bincount(
            z_sample.astype(int),
            minlength=len(candidate_angles),
        )

    if label_counts.sum() > 0:
        label_weights = label_counts / label_counts.sum()
    else:
        label_weights = label_counts

    plt.figure(figsize=(10, 4))

    plt.bar(
        candidate_angles,
        true_weights,
        width=4.0,
        alpha=0.35,
        label="Ground truth",
    )

    plt.plot(
        candidate_angles,
        posterior_pi_mean,
        linewidth=1.5,
        label="Posterior mean DP weights",
    )

    plt.plot(
        candidate_angles,
        label_weights,
        linewidth=1.2,
        linestyle="--",
        label="Posterior mean label frequencies",
    )

    plt.axvline(TRUE_ANGLE_LOW, linestyle=":", linewidth=1)
    plt.axvline(TRUE_ANGLE_HIGH, linestyle=":", linewidth=1)

    plt.xlabel("Angle (degrees)")
    plt.ylabel("Probability")
    plt.title(title)
    plt.xlim(0.0, 360.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        Path(out_dir) / "true_vs_posterior_angle_distribution.png",
        dpi=150,
    )
    plt.close()


def save_reconstruction_summary(xs, out_dir, title):
    xs = np.asarray(xs)

    x_last = xs[-1]
    x_mean = xs.mean(axis=0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("True image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(x_last, cmap="gray")
    plt.title("Last sample")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(x_mean, cmap="gray")
    plt.title("Posterior mean")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(
        Path(out_dir) / "reconstruction_summary.png",
        dpi=150,
    )
    plt.close()


# --------------------------------------------------
# Vanilla Gibbs with DP(alpha, Uniform) angle prior
# --------------------------------------------------

for seed in seeds:
    out_dir = output_root / (
        f"vanilla_gibbs_em_init_alpha{int(alpha_angle)}_seed{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning Vanilla Gibbs, seed={seed}")

    xs_gibbs, zs_gibbs, pis_gibbs = (
        vanilla_gibbs_sampler_angle_prior(
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
            dir=str(out_dir),
        )
    )

    np.save(out_dir / "xs_gibbs.npy", np.asarray(xs_gibbs))
    np.save(out_dir / "zs_gibbs.npy", np.asarray(zs_gibbs))
    np.save(out_dir / "pis_angle_gibbs.npy", np.asarray(pis_gibbs))

    save_angle_distribution_comparison(
        pis_gibbs,
        zs_gibbs,
        out_dir,
        "Vanilla Gibbs: true vs inferred angle distribution",
    )

    save_reconstruction_summary(
        xs_gibbs,
        out_dir,
        "Vanilla Gibbs with truncated-angle data",
    )

    print("Saved Vanilla Gibbs output to:", out_dir)


# --------------------------------------------------
# Gibbs-LD with DP(alpha, Uniform) angle prior
# --------------------------------------------------

for seed in seeds:
    out_dir = output_root / (
        f"gibbs_ld_em_init_alpha{int(alpha_angle)}_seed{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning Gibbs-LD, seed={seed}")

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
        dir=str(out_dir),
    )

    np.save(out_dir / "xs_ld.npy", np.asarray(xs_ld))
    np.save(out_dir / "zs_ld.npy", np.asarray(zs_ld))
    np.save(out_dir / "pis_angle_ld.npy", np.asarray(pis_ld))

    save_angle_distribution_comparison(
        pis_ld,
        zs_ld,
        out_dir,
        "Gibbs-LD: true vs inferred angle distribution",
    )

    save_reconstruction_summary(
        xs_ld,
        out_dir,
        "Gibbs-LD with truncated-angle data",
    )

    print("Saved Gibbs-LD output to:", out_dir)


# --------------------------------------------------
# MALA with DP(alpha, Uniform) angle prior
# --------------------------------------------------

for seed in seeds:
    out_dir = output_root / (
        f"mala_em_init_alpha{int(alpha_angle)}_seed{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning MALA, seed={seed}")

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
        dir=str(out_dir),
    )

    np.save(out_dir / "xs_mala.npy", np.asarray(xs_mala))
    np.save(out_dir / "zs_mala.npy", np.asarray(zs_mala))
    np.save(out_dir / "pis_angle_mala.npy", np.asarray(pis_mala))

    save_angle_distribution_comparison(
        pis_mala,
        zs_mala,
        out_dir,
        "MALA: true vs inferred angle distribution",
    )

    save_reconstruction_summary(
        xs_mala,
        out_dir,
        "MALA with truncated-angle data",
    )

    print("Saved MALA output to:", out_dir)
