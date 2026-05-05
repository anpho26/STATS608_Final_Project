# In root dir, run: python3 -m scripts.run_MCMC_single

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib import pyplot as plt

from src.utils import load_image, circle_mask, simulate_data, \
                      generate_i, generate_square, generate_Sshape
from src.experiment import MCMC_experiment

d = 32
n = 1000
seed1 = 345
seed2 = 100
sigma2 = 0.5

candidate_angles = np.arange(0, 180, 4, dtype=float)

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

MCMC_experiment(d=d, img=img, data=data, seed=seed2,
                candidate_angles=candidate_angles, sigma2=sigma2, save_dir='exp_output/test1')