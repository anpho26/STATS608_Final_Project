#!/usr/bin/env python
# coding: utf-8

# In[17]:


import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import PCA
from scipy.ndimage import rotate

# go up one level: notebooks/ → project/
sys.path.append(str(Path().resolve().parent))


# In[2]:


from src.utils import load_image, circle_mask, simulate_data, \
                      generate_i, generate_square, generate_Sshape
from src.em import em_algorithm
from src.gibbs import vanilla_gibbs_sampler, gibbs_LD_sampler, pack_image
from src.metrics import measure_projection_error, \
                        measure_projection_error_minimal, \
                        measure_alignment_error, \
                        measure_projection_error_batch, \
                        measure_alignment_error_batch


# In[3]:


# Two ground truth choices
d = 32
mask_d = circle_mask(d)
img0 = generate_Sshape(size=d)
img0 = np.where(mask_d, img0, 0.0)
img1 = load_image('../images/phantom.jpeg', size=d)
img1 = np.where(mask_d, img1, 0.0)

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
axes[0].imshow(img0, cmap='gray')
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].imshow(img1, cmap='gray')
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.tight_layout()
plt.show()


# In[4]:


d = 32
n = 1000
seed1 = 345
seed2 = 100
sigma2 = 0.5

candidate_angles = np.arange(0, 180, 4, dtype=float)

mask_d = circle_mask(d)
img = load_image('../images/phantom.jpeg', size=d)
img = np.where(mask_d, img, 0.0)

data, true_angles = simulate_data(
    img,
    candidate_angles,
    n_obs=n,
    noise_std=np.sqrt(sigma2),
    seed=seed1,
)

xs_em = em_algorithm(data, candidate_angles,
                     n_em=200, n_inner=50, lr=1e-4, lam=5e-3,
                     temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                     seed=0, sigma2=None, verbose=10, x_init=None)

xs_em_refined = em_algorithm(data, candidate_angles,
                             n_em=300, n_inner=50, lr=1e-5, lam=5e-3,
                             temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                             seed=0, sigma2=sigma2, verbose=10, x_init=xs_em[-1])


# In[5]:


vg_samples_xs, vg_samples_zs = vanilla_gibbs_sampler(data, candidate_angles, sigma2, sigma_eps2=1, n_samples=1000, n_burnins=100,
                                                     random_state=seed2, x_init=pack_image(xs_em_refined[-1], d), verbose=10)


# In[6]:


ld_samples_xs, ld_samples_zs = gibbs_LD_sampler(data, candidate_angles,
                                                n_gibbs=1000, n_burnins=200, n_inner=50, lr=1e-5, lam=5e-2,
                                                temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                                                seed=0, sigma2=sigma2, verbose=10, x_init=xs_em_refined[-1])


# In[7]:


proj_error_em = measure_projection_error_minimal(xs_em_refined[-1], data, candidate_angles)
algm_error_em = measure_alignment_error(xs_em_refined[-1], img, candidate_angles)
f"Proj. error: {float(proj_error_em):.4f}; algm. error: {float(algm_error_em):.4f}"


# In[14]:


# Calculate errors
proj_errors_vg = measure_projection_error_batch(vg_samples_xs, vg_samples_zs, data)
algm_errors_vg = measure_alignment_error_batch(vg_samples_xs, img, candidate_angles)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
axes[0].hist(proj_errors_vg, bins=20)
axes[0].set_title(f"Proj. error. Mean {float(np.mean(proj_errors_vg)):.4f}", fontsize=12)
axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axes[1].hist(algm_errors_vg, bins=20)
axes[1].set_title(f"Algm. error. Mean {float(np.mean(algm_errors_vg)):.4f}", fontsize=12)
axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.suptitle("Errors for vanilla Gibbs sampler", fontsize=12)
plt.tight_layout()
# plt.savefig('../output/errors_VG_phantom.png')
plt.show()


# In[15]:


# Calculate errors
ld_samples_zs_ = [candidate_angles[z].copy() for z in ld_samples_zs]
proj_errors_ld = measure_projection_error_batch(ld_samples_xs, ld_samples_zs_, data)
algm_errors_ld = measure_alignment_error_batch(ld_samples_xs, img, candidate_angles)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
axes[0].hist(proj_errors_ld, bins=20)
axes[0].set_title(f"Proj. error. Mean {float(np.mean(proj_errors_ld)):.4f}", fontsize=12)
axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axes[1].hist(algm_errors_ld, bins=20)
axes[1].set_title(f"Algm. error. Mean {float(np.mean(algm_errors_ld)):.4f}", fontsize=12)
axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.suptitle("Errors for Gibbs - LD sampler", fontsize=12)
plt.tight_layout()
# plt.savefig('../output/errors_LD_phantom.png')
plt.show()


# In[10]:


# Calculate label switching
def count_switch(zs):
    n = len(zs)
    k = len(zs[0])
    count = 0
    for i in range(n-1):
        count += k-np.isclose(0., zs[i+1]-zs[i]).astype(int).sum()
    return count

n_samples=1000
print(f"Label switch count for vanilla Gibbs: {count_switch(vg_samples_zs)}, in prop, {(count_switch(vg_samples_zs)/(n_samples-1)/n):.4f}")
print(f"Label switch count for Gibbs LD: {count_switch(ld_samples_zs)}, in prop, {(count_switch(ld_samples_zs)/(n_samples-1)/n):.4f}")


# In[16]:


# Compute means
vg_mean = np.mean(np.array(vg_samples_xs), axis=0)
ld_mean = np.mean(np.array(ld_samples_xs), axis=0)

# Compute distances vg
k = len(candidate_angles)
dists = np.abs(vg_samples_zs[-1][:, None] - candidate_angles[None, :])
idx = np.argmin(dists, axis=1)
mask = np.isclose(vg_samples_zs[-1], candidate_angles[idx], atol=1e-6)
idx = idx[mask]
vg_counts = np.bincount(idx, minlength=k)

# Compute distances ld
ld_counts = np.bincount(ld_samples_zs[-1], minlength=k)


fig, axes = plt.subplots(2, 4, figsize=(14, 7))
vmin = -1.
vmax = np.max(img)+1.

axes[0][0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])
axes[0][0].set_title("Ground truth")

axes[1][0].imshow(xs_em_refined[-1], cmap='gray', vmin=vmin, vmax=vmax)
axes[1][0].set_xticks([])
axes[1][0].set_yticks([])
axes[1][0].set_title("EM output")

axes[0][1].imshow(vg_samples_xs[-1], cmap='gray', vmin=vmin, vmax=vmax)
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])
axes[0][1].set_title("VG last")

axes[1][1].imshow(ld_samples_xs[-1], cmap='gray', vmin=vmin, vmax=vmax)
axes[1][1].set_xticks([])
axes[1][1].set_yticks([])
axes[1][1].set_title("GLD last")

axes[0][2].imshow(vg_mean, cmap='gray', vmin=vmin, vmax=vmax)
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])
axes[0][2].set_title("VG mean")

axes[1][2].imshow(ld_mean, cmap='gray', vmin=vmin, vmax=vmax)
axes[1][2].set_xticks([])
axes[1][2].set_yticks([])
axes[1][2].set_title("GLD mean")

axes[0][3].bar(list(range(k)), vg_counts, width=1.0)
axes[0][3].set_xticks([])
axes[0][3].set_title("VG last assignments")

axes[1][3].bar(list(range(k)), ld_counts, width=1.0)
axes[1][3].set_xticks([])
axes[1][3].set_title("GLD last assignments")

plt.tight_layout()
# plt.savefig('../output/images_phantom.png')
plt.show()


# In[34]:


def pca_embed_images(images, n_components=2, normalize=True):
    """
    images: list or array of shape (n, d, d)
    returns: array of shape (n, n_components)
    """
    X = np.asarray(images)
    n = X.shape[0]

    # Flatten images to (n, d*d)
    X = X.reshape(n, -1)

    # Optional per-image normalization
    if normalize:
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mean) / std

    # PCA
    pca = PCA(n_components=n_components)
    X_embedded = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)

    return X_embedded


# In[19]:


img_spin = img.copy()
dist = np.inf
for ang in candidate_angles:
    img_temp = rotate(img, angle=ang, reshape=False)
    dist_temp = np.linalg.norm(img_temp-xs_em_refined[-1])
    if dist_temp < dist:
        dist = dist_temp
        img_spin = img_temp


# In[20]:


plt.imshow(img_spin)


# In[35]:


n_samples = 1000
classes = np.array([0, 1] + [2 for _ in range(n_samples)] + [3 for _ in range(n_samples//4)] + [4 for _ in range(n_samples//4)] + \
                   [5 for _ in range(n_samples//4)] + [6 for _ in range(n_samples//4)])
imgs_all = [img_spin, xs_em_refined[-1]] + vg_samples_xs + ld_samples_xs
imgs_all_encode = pca_embed_images(imgs_all, n_components=2, normalize=False)


# In[36]:


names = ['True', 'EM', 'VG', 'LDG1', 'LDG2', 'LDG3', 'LDG4']
colors = ['C1', 'C3', 'C2', '#ADD7F6', '#87BFFF', '#3F8EFC', '#2667FF']
for i, (name, color) in enumerate(zip(names, colors)):
    taken = (classes == i)
    plt.scatter(imgs_all_encode[taken][:, 0], imgs_all_encode[taken][:, 1], c=color, label=name, s=10, alpha=0.7, zorder=10-i)
plt.title('PCA of true image and generated estimates - phantom')
plt.legend()
# plt.savefig('../output/plot_PCA_phantom.png')
plt.show()


# In[ ]:




