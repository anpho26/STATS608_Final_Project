# In root dir, run:
# python3 -m scripts.run_mixture2_Gibbs

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from src.utils import (
    load_image,
    circle_mask,
    simulate_mixture_data,
    make_symbol,
)

from src.gibbs_mixture import (
    vanilla_gibbs_mixture_sampler,
    MALA_mixture_sampler,
)


# -----------------------------
# Basic settings
# -----------------------------

d = 32
n = 1000
seed_data = 345
sigma2 = 0.5
sigma_eps2 = sigma2

candidate_angles = np.arange(0, 360, 4, dtype=float)

seeds = [100, 200, 300]


# -----------------------------
# Generate / load two images
# -----------------------------

mask_d = circle_mask(d)

# Option 1: use two symbols
img_1 = make_symbol("4", size=d, angle=20)
img_2 = make_symbol("?", size=d, angle=-20)

# Optional circular mask
# You may want to comment this out for symbol images.
# img_1 = np.where(mask_d, img_1, 0.0)
# img_2 = np.where(mask_d, img_2, 0.0)

# Quick check
fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
axes[0].imshow(img_1, cmap="gray")
axes[0].set_title("Image 1")
axes[0].axis("off")

axes[1].imshow(img_2, cmap="gray")
axes[1].set_title("Image 2")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("exp_output/mixture2_true_images.png", dpi=150)
plt.close()


# -----------------------------
# Simulate mixture data
# -----------------------------

data, true_classes, true_angles = simulate_mixture_data(
    image1=img_1,
    image2=img_2,
    n_obs=n,
    noise_std=np.sqrt(sigma2),
    pi=(0.5, 0.5),
    seed=seed_data,
    angle_low=0.0,
    angle_high=360.0,
)

print("Data shape:", data.shape)
print("True class counts:", np.bincount(true_classes)[1:])
print("True angle range:", true_angles.min(), true_angles.max())


# -----------------------------
# Run vanilla Gibbs mixture
# -----------------------------

for seed in seeds:
    print(f"\nRunning vanilla Gibbs mixture, seed={seed}")

    out_dir = f"exp_output/mixture2_gibbs_seed{seed}"

    xs_gibbs, zs_gibbs, cs_gibbs, pis_gibbs = vanilla_gibbs_mixture_sampler(
        data=data,
        candidate_angles=candidate_angles,
        sigma2=sigma2,
        sigma_eps2=sigma_eps2,
        alpha=1.0,
        n_mixtures=2,
        n_samples=200,
        n_burnins=50,
        random_state=seed,
        x_init=None,
        verbose=10,
        imshow=False,
        imsave=True,
        dir=out_dir,
    )

    np.save(f"{out_dir}/xs_gibbs.npy", np.array(xs_gibbs))
    np.save(f"{out_dir}/zs_gibbs.npy", np.array(zs_gibbs))
    np.save(f"{out_dir}/cs_gibbs.npy", np.array(cs_gibbs))
    np.save(f"{out_dir}/pis_gibbs.npy", np.array(pis_gibbs))

    print(f"Saved Gibbs mixture output to {out_dir}")


# -----------------------------
# Run MALA mixture
# -----------------------------

for seed in seeds:
    print(f"\nRunning MALA mixture, seed={seed}")

    out_dir = f"exp_output/mixture2_mala_seed{seed}"

    xs_mala, zs_mala, cs_mala, pis_mala = MALA_mixture_sampler(
        data=data,
        candidate_angles=candidate_angles,
        sigma2=sigma2,
        alpha=1.0,
        n_mixtures=2,
        n_gibbs=200,
        n_burnins=50,
        n_inner=20,
        lr=1e-5,
        lam=5e-3,
        temp_start=2.0,
        temp_end=1.0,
        temp_decay=0.995,
        seed=seed,
        x_init=None,
        verbose=10,
        imshow=False,
        imsave=True,
        dir=out_dir,
    )

    np.save(f"{out_dir}/xs_mala.npy", np.array(xs_mala))
    np.save(f"{out_dir}/zs_mala.npy", np.array(zs_mala))
    np.save(f"{out_dir}/cs_mala.npy", np.array(cs_mala))
    np.save(f"{out_dir}/pis_mala.npy", np.array(pis_mala))

    print(f"Saved MALA mixture output to {out_dir}")