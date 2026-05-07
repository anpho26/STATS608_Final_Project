# In root dir, run: python3 -m scripts.run_MCMC_single

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib import pyplot as plt

from src.utils import load_image, circle_mask, simulate_data, \
                      generate_i, generate_square, generate_Sshape
from src.experiment import MCMC_experiment

d = 4
n = 1000
seed1 = 345
sigma2 = 0.5

candidate_angles = np.arange(0, 180, 4, dtype=float)



# run for S-shaped image


mask_d = circle_mask(d)
img = generate_Sshape(size=d)
img = np.where(mask_d, img, 0.0)

data, true_angles = simulate_data(
    img,
    candidate_angles,
    n_obs=n,
    noise_std=np.sqrt(sigma2),
    seed=seed1,
)

seeds = [100, 200, 300]

for seed in seeds:
    MCMC_experiment(
        d=d,
        img=img,
        data=data,
        seed=seed,
        candidate_angles=candidate_angles,
        sigma2=sigma2,
        save_dir=f'exp_output/test1_seed{seed}'
    )

# run MCMC for random-signal S-shaped image

img_2 = generate_Sshape(size=d)
img_2 = np.where(mask_d, img_2, 0.0)

data_2, true_angles_2 = simulate_data(
    img_2,
    candidate_angles,
    n_obs=n,
    noise_std=np.sqrt(sigma2),
    seed=seed1,
)

seeds = [100, 200, 300]

for seed in seeds:
    MCMC_experiment(
        d=d,
        img=img_2,
        data=data_2,
        seed=seed,
        candidate_angles=candidate_angles,
        sigma2=sigma2,
        save_dir=f'exp_output/test2_seed{seed}'
    )
