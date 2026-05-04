#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import PCA
from scipy.ndimage import rotate

# go up one level: notebooks/ → project/
sys.path.append(str(Path().resolve().parent))


# In[13]:


from src.utils import load_image, circle_mask, simulate_data, simulate_mixture_data, \
                      generate_i, generate_square, generate_Sshape, radon_rows
from src.em import em_algorithm, em_algorithm_2classes
from src.gibbs import vanilla_gibbs_sampler, gibbs_LD_sampler, pack_image
from src.gibbs_mixture import vanilla_gibbs_mixture_sampler
from src.metrics import measure_projection_error, \
                        measure_projection_error_minimal, \
                        measure_alignment_error, \
                        measure_projection_error_batch, \
                        measure_alignment_error_batch


# In[3]:


# # Ground truth choices
# d = 32
# img = generate_Sshape(size=d)
# img = generate_i(size=d)
# img = generate_square(size=d)
# img = load_image('../images/phantom.jpeg', size=d)


# In[4]:


d = 32
n = 1000
seed1 = 345
seed2 = 123
sigma2 = 0.5

candidate_angles = np.arange(0, 180, 4, dtype=float)

mask_d = circle_mask(d)
img1 = generate_i(size=d)
img1 = np.where(mask_d, img1, 0.0)
img2 = generate_Sshape(size=d)
img2 = np.where(mask_d, img2, 0.0)

data, true_classes, true_angles = simulate_mixture_data(
    img1, img2,
    candidate_angles,
    n_obs=n,
    noise_std=np.sqrt(sigma2),
    seed=seed1,
)

x1s_em, x2s_em, pis_em = em_algorithm_2classes(data, candidate_angles,
                                               n_em=400, n_inner=50, lr=1e-4, lam=5e-3,
                                               temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                                               seed=seed2, sigma2=None, verbose=10,
                                               x_init1=None, x_init2=None, pi_init=0.5)

x1s_em_refined, x2s_em_refined, pis_em_refined = em_algorithm_2classes(data, candidate_angles,
                                                                       n_em=600, n_inner=50, lr=1e-5, lam=5e-3,
                                                                       temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                                                                       seed=seed2, sigma2=sigma2, verbose=10,
                                                                       x_init1=x1s_em[-1], x_init2=x2s_em[-1], pi_init=pis_em[-1])


# In[5]:


x_init = np.array([x1s_em_refined[-1], x2s_em_refined[-1]])
xs_vg, zs_vg, cs_vg, pis_vg = vanilla_gibbs_mixture_sampler(data, candidate_angles, sigma2=1., sigma_eps2=sigma2,
                                                            alpha=[3.], n_mixtures=2, n_samples=1000, n_burnins=100,
                                                            random_state=None, x_init=x_init, verbose=10)


# In[37]:


# Extensions to mixtures of images
def measure_projection_error_multiclass(x, z, c, data):
    m = len(x)
    losses = []
    for i in range(m):
        taking = (c==i)
        if len(taking) > 0: losses.append(measure_projection_error(x[i], z[taking], data[taking])**2*len(taking))
    return np.sqrt(np.sum(np.array(losses))/len(c))

def measure_projection_error_minimal_multiclass(x, data, candidate_angles): # Needs work
    projs = np.vstack([radon_rows(x_, candidate_angles) for x_ in x])
    dists = np.mean((data[:, None, :]-projs[None, :, :])**2, axis=2)
    dists = dists.min(axis=1)
    return np.sqrt(np.mean(dists))

def measure_alignment_error_multiclass(x, x_true, candidates_angles):
    return np.mean(np.array([measure_alignment_error(x_, x_true_, candidates_angles) for x_, x_true_ in zip(x, x_true)]))

def measure_projection_error_batch_multiclass(xs, zs, cs, data):
    errs = []
    for x, z, c in zip(xs, zs, cs):
        err = measure_projection_error_multiclass(x, z, c, data)
        errs.append(err)
    return np.array(errs)

def measure_alignment_error_batch_multiclass(xs, x_true, candidates_angles):
    errs = []
    for x in xs:
        err = measure_alignment_error_multiclass(x, x_true, candidates_angles)
        errs.append(err)
    return np.array(errs)


# In[25]:


measure_alignment_error(x_init[0], img2f, candidate_angles)


# In[26]:


img2f = np.fliplr(img2)
imgs1 = np.array([img1, img2f])
imgs2 = np.array([img2f, img1])
proj_error_em = measure_projection_error_minimal_multiclass(x_init, data, candidate_angles)
algm_error_em = min(measure_alignment_error_multiclass(x_init, imgs1, candidate_angles), 
                    measure_alignment_error_multiclass(x_init, imgs2, candidate_angles))
f"Proj. error: {float(proj_error_em):.4f}; algm. error: {float(algm_error_em):.4f}"


# In[46]:


# Calculate errors
proj_errors_vg = measure_projection_error_batch_multiclass(xs_vg, zs_vg, cs_vg, data)
algm_errors_vg = measure_alignment_error_batch_multiclass(xs_vg, imgs2, candidate_angles)


# In[61]:


# Plotting
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
axes[0].hist(proj_errors_vg, bins=20)
axes[0].set_title(f"Proj. error. Mean {float(np.mean(proj_errors_vg)):.4f}", fontsize=12)
axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axes[1].hist(algm_errors_vg, bins=20)
axes[1].set_title(f"Algm. error. Mean {float(np.mean(algm_errors_vg)):.4f}", fontsize=12)
axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axes[2].hist([pi[0] for pi in pis_vg], bins=20)
axes[2].set_title(f"Dist. of proportion", fontsize=12)
axes[2].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.suptitle("Errors for Vanilla Gibbs sampler", fontsize=12)
plt.tight_layout()
# plt.savefig('../output/errors_mixture.png')
plt.show()


# In[57]:


# Compute means
vg_mean1 = np.mean(np.array([x[0] for x in xs_vg]), axis=0)
vg_mean2 = np.mean(np.array([x[1] for x in xs_vg]), axis=0)


fig, axes = plt.subplots(2, 4, figsize=(14, 7))
vmin = -1.
vmax = max(np.max(img1), np.max(img2)) + 1.

axes[0][0].imshow(img1, cmap='gray', vmin=vmin, vmax=vmax)
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])
axes[0][0].set_title("Ground truth 1")

axes[1][0].imshow(img2, cmap='gray', vmin=vmin, vmax=vmax)
axes[1][0].set_xticks([])
axes[1][0].set_yticks([])
axes[1][0].set_title("Ground truth 2")

axes[0][1].imshow(x2s_em_refined[-1], cmap='gray', vmin=vmin, vmax=vmax)
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])
axes[0][1].set_title("EM 1")

axes[1][1].imshow(x1s_em_refined[-1], cmap='gray', vmin=vmin, vmax=vmax)
axes[1][1].set_xticks([])
axes[1][1].set_yticks([])
axes[1][1].set_title("EM 2")

axes[0][2].imshow(xs_vg[-1][1], cmap='gray', vmin=vmin, vmax=vmax)
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])
axes[0][2].set_title("VG last 1")

axes[1][2].imshow(xs_vg[-1][0], cmap='gray', vmin=vmin, vmax=vmax)
axes[1][2].set_xticks([])
axes[1][2].set_yticks([])
axes[1][2].set_title("VG last 2")

axes[0][3].imshow(vg_mean2, cmap='gray', vmin=vmin, vmax=vmax)
axes[0][3].set_xticks([])
axes[0][3].set_yticks([])
axes[0][3].set_title("VG mean 1")

axes[1][3].imshow(vg_mean1, cmap='gray', vmin=vmin, vmax=vmax)
axes[1][3].set_xticks([])
axes[1][3].set_yticks([])
axes[1][3].set_title("VG mean 2")

plt.tight_layout()
# plt.savefig('../output/images_mixture.png')
plt.show()


# In[7]:


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

    return X_embedded


# In[9]:


img_spin = img.copy()
dist = np.inf
for ang in candidate_angles:
    img_temp = rotate(img, angle=ang, reshape=False)
    dist_temp = np.linalg.norm(img_temp-xs_em_refined[-1])
    if dist_temp < dist:
        dist = dist_temp
        img_spin = img_temp
    


# In[ ]:




