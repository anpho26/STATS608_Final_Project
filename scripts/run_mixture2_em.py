import numpy as np
import matplotlib.pyplot as plt

from cryo_em_em.radon_ops import circle_mask
from cryo_em_em.shapes import make_diagonal_stick
from cryo_em_em.simulate import simulate_mixture_data
from cryo_em_em.em_mixture import em_reconstruct_mixture2


d = 32
candidate_angles = np.arange(0, 180, 4, dtype=float)
mask = circle_mask(d)

img1 = make_diagonal_stick(size=d, width=1)
img2 = make_diagonal_stick(size=d, width=5)

img1 = img1 * (img2.sum() / img1.sum())

img1 = np.where(mask, img1, 0.0)
img2 = np.where(mask, img2, 0.0)

Y, true_classes, true_angles = simulate_mixture_data(
    img1,
    img2,
    candidate_angles,
    n_obs=500,
    noise_std=0.01,
    pi=(0.5, 0.5),
    seed=123,
)

out = em_reconstruct_mixture2(
    Y,
    candidate_angles,
    output_size=d,
    true_classes=true_classes,
    true_angles=true_angles,
    true_img1=img1,
    true_img2=img2,
    n_em=500,
    n_inner=30,
    lr=1e-4,
    lam=5e-3,
    temp_start=2.0,
    temp_end=0.3,
    temp_decay=0.99,
    seed=123,
    verbose=True,
    verbose_tqdm=True,
    show_every=10,
)

print("Final objective:", out["obj_hist"][-1])
print("Final projection error:", out["proj_err_hist"][-1])
print("Final aligned mixture error:", out["align_err_hist"][-1])
print("Final matching:", out["align_match_hist"][-1])
print("Estimated pi:", out["pi"])

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].imshow(img1, cmap="gray")
axes[0].set_title("True image 1")
axes[0].axis("off")

axes[1].imshow(img2, cmap="gray")
axes[1].set_title("True image 2")
axes[1].axis("off")

axes[2].imshow(out["x1"], cmap="gray")
axes[2].set_title("Estimated image 1")
axes[2].axis("off")

axes[3].imshow(out["x2"], cmap="gray")
axes[3].set_title("Estimated image 2")
axes[3].axis("off")

plt.tight_layout()
plt.show()