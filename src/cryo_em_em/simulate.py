import numpy as np
from .radon_ops import radon_rows


def simulate_data(image, candidate_angles, n_obs=500, noise_std=0.01, seed=0):
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, len(candidate_angles), size=n_obs)
    true_angles = candidate_angles[idx]

    clean = radon_rows(image, true_angles)
    Y = clean + noise_std * rng.standard_normal(clean.shape)

    return Y, true_angles


def simulate_mixture_data(
    img1,
    img2,
    candidate_angles,
    n_obs=500,
    noise_std=0.01,
    pi=(0.5, 0.5),
    seed=0,
):
    rng = np.random.default_rng(seed)

    z = rng.choice(2, size=n_obs, p=pi)
    angle_idx = rng.integers(0, len(candidate_angles), size=n_obs)
    true_angles = candidate_angles[angle_idx]

    Y = []
    for i in range(n_obs):
        img = img1 if z[i] == 0 else img2
        y = radon_rows(img, [true_angles[i]])[0]
        y = y + noise_std * rng.standard_normal(y.shape)
        Y.append(y)

    Y = np.asarray(Y)
    true_classes = z + 1

    return Y, true_classes, true_angles