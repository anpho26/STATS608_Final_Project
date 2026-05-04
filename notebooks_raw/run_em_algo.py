#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# go up one level: notebooks/ → project/
sys.path.append(str(Path().resolve().parent))


# In[2]:


from src.utils import load_image, circle_mask, simulate_data, \
                      generate_i, generate_square, generate_Sshape
from src.em import em_algorithm


# In[3]:


d = 32
n = 1000
seed1 = 345
seed2 = 100
sigma2 = 0.5

candidate_angles = np.arange(0, 180, 4, dtype=float)

mask_d = circle_mask(d)
img = generate_i(size=d)
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


# In[7]:


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


# In[10]:


mask_d = circle_mask(d)
img0 = generate_i(size=d)
img0 = np.where(mask_d, img0, 0.0)
img1 = load_image('../images/phantom.jpeg', size=d)
img1 = np.where(mask_d, img, 0.0)

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
axes[0].imshow(img0, cmap='gray')
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].imshow(img1, cmap='gray')
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.tight_layout()
plt.show()


# In[ ]:




