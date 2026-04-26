import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.special import logsumexp
from tqdm.auto import tqdm
from IPython.display import clear_output, display
from skimage.transform import iradon

from utils import circle_mask, radon_rows

# Helper functions for the EM algorithm
def backproject_single(proj, angle, output_size):
    sino = proj[:, None]
    return iradon(
        sino,
        theta=[angle],
        filter_name=None,
        circle=True,
        output_size=output_size,
    )

def simulate_data(image, candidate_angles, n_obs=60, noise_std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(candidate_angles), size=n_obs)
    true_angles = candidate_angles[idx]
    clean = radon_rows(image, true_angles).T
    Y = clean + noise_std * rng.standard_normal(clean.shape)
    return Y, true_angles

def random_init(size, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((size, size))
    x = (x + np.flipud(x)) / 2.0
    x -= x.min()
    if x.max() > 0:
        x /= x.max()
    return x

# Main EM algorithm
def em_algorithm(data, candidate_angles, 
                 n_em=100, n_inner=50, lr=1e-4, lam=5e-3,
                 temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                 seed=0, sigma2=None, verbose=-1, x_init=None):
    
    # Set ups
    n, d = data.shape
    k = len(candidate_angles)
    mask = circle_mask(d)
    x = random_init(d, seed=seed) if x_init is None else x_init.copy()
    x = np.where(mask, x, 0.0)
    if sigma2 is None: sigma2 = np.var(data)
    pbar = tqdm(range(n_em), desc="EM iterations")
    xs = [x.copy()]
    plot_x = list(range(k))
    plt.imshow(x)
    plt.show()

    # Main loop
    for em_it in pbar:
        temperature = max(temp_end, temp_start * (temp_decay ** em_it))

        # E-step
        pred = radon_rows(x, candidate_angles)
        d2 = np.sum((data[:, None, :] - pred[None, :, :]) ** 2, axis=2)
        log_resp = -0.5 * d2 / (temperature * sigma2)
        log_resp -= log_resp.max(axis=1, keepdims=True)
        R = np.exp(log_resp)
        R /= R.sum(axis=1, keepdims=True)
        B = R.T @ data
        counts = R.sum(axis=0)

        # M-step
        for _ in range(n_inner):
            pred = radon_rows(x, candidate_angles)
            grad = np.zeros_like(x)

            # Gradient ...
            for m, ang in enumerate(candidate_angles):
                if counts[m] < 1e-12:
                    continue
                resid = counts[m] * pred[m] - B[m]
                grad += backproject_single(resid, ang, d)

            # ... decend
            grad = grad / sigma2 + lam * x
            x = x - lr * grad
            x = np.clip(x, 0.0, None)
            x = np.where(mask, x, 0.0)

        # Storate
        xs.append(x.copy())

        # Showing progress
        if verbose >= 0:
            if ((em_it + 1) % verbose == 0) or (em_it == 0):
                print(f"EM iter {em_it+1:04d} | "
                      f"temp={temperature:.4f} | "
                      f"variance={sigma2:.4f} ")
                fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
                axes[0].imshow(xs[-1], cmap='gray')
                axes[0].set_title("Sample")
                axes[0].set_xticks([])
                axes[0].set_yticks([])
                axes[1].bar(plot_x, counts)
                axes[1].set_title("Responsibilities")
                axes[1].set_xticks([])
                plt.tight_layout()
                plt.show()

    return xs
