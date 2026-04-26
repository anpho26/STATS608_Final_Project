from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cryo_em_em.io import load_image
from cryo_em_em.radon_ops import circle_mask
from cryo_em_em.simulate import simulate_data
from cryo_em_em.em_single import em_reconstruct_live


ROOT = Path(__file__).resolve().parents[1]
PHANTOM_PATH = ROOT / "data" / "raw" / "fstPhantom.png"

d = 32
n = 1000
sigma2 = 0.5
seed1 = 345
seed2 = 100

candidate_angles = np.arange(0, 180, 4, dtype=float)

mask = circle_mask(d)
img = load_image(PHANTOM_PATH, size=d)
img = np.where(mask, img, 0.0)

Y, true_angles = simulate_data(
    img,
    candidate_angles,
    n_obs=n,
    noise_std=np.sqrt(sigma2),
    seed=seed1,
)

xs, est_angles, R, metrics = em_reconstruct_live(
    Y,
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
    seed=seed2,
    verbose=True,
    verbose_tqdm=True,
    show_every=10,
)

recon = xs[-1]

print("Final objective:", metrics["obj_hist"][-1])
print("Final projection error:", metrics["proj_err_hist"][-1])
print("Final aligned error:", metrics["align_err_hist"][-1])

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("True image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(recon, cmap="gray")
plt.title("EM reconstruction")
plt.axis("off")

plt.tight_layout()
plt.show()