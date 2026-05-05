import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.special import logsumexp
from tqdm.auto import tqdm
from IPython.display import clear_output, display
from skimage.transform import iradon

from src.utils import circle_mask, radon_rows, random_init, backproject_single

# Main EM algorithm
def em_algorithm(data, candidate_angles, 
                 n_em=100, n_inner=50, lr=1e-4, lam=5e-3,
                 temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                 seed=0, sigma2=None, verbose=-1, x_init=None,
                 imshow=True, imsave=False, dir='plotsEM'):
    
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
    os.makedirs(dir, exist_ok=True)

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
                title=f"EM iter {em_it+1:04d} | "+\
                      f"temp={temperature:.4f} | "+\
                      f"variance={sigma2:.4f} "
                fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
                axes[0].imshow(xs[-1], cmap='gray')
                axes[0].set_title("Sample")
                axes[0].set_xticks([])
                axes[0].set_yticks([])
                axes[1].bar(plot_x, counts, width=1.0, linewidth=0)
                axes[1].set_title("Responsibilities")
                axes[1].set_xticks([])
                plt.tight_layout()
                plt.suptitle(title)
                if imsave:
                    plt.savefig(f'{dir}/emIter{em_it}.png')
                if imshow: plt.show()
                plt.close()

    return xs

# Main EM algorithm
def em_algorithm_2classes(data, candidate_angles, 
                          n_em=100, n_inner=50, lr=1e-4, lam=5e-3,
                          temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                          seed=0, sigma2=None, verbose=-1,
                          x_init1=None, x_init2=None, pi_init=0.5,
                          imshow=True, imsave=False, dir='plotsEM2c'):
    
    # Set ups
    n, d = data.shape
    k = len(candidate_angles)
    mask = circle_mask(d)
    x1 = random_init(d, seed=seed) if x_init1 is None else x_init1.copy()
    x1 = np.where(mask, x1, 0.0)
    x2 = random_init(d, seed=seed+1) if x_init2 is None else x_init2.copy()
    x2 = np.where(mask, x2, 0.0)
    pi = np.random.beta(a=2., b=2.) if pi_init is None else pi_init
    if sigma2 is None: sigma2 = np.var(data)
    pbar = tqdm(range(n_em), desc="EM iterations")
    x1s = [x1.copy()]
    x2s = [x2.copy()]
    pis = [pi]
    plot_x = list(range(k))
    os.makedirs(dir, exist_ok=True)

    # Main loop
    for em_it in pbar:
        temperature = max(temp_end, temp_start * (temp_decay ** em_it))

        # E-step
        pred1 = radon_rows(x1, candidate_angles)
        pred2 = radon_rows(x2, candidate_angles)
        d2_1 = np.sum((data[:, None, :] - pred1[None, :, :]) ** 2, axis=2)
        d2_2 = np.sum((data[:, None, :] - pred2[None, :, :]) ** 2, axis=2)
        log_r1 = np.log(pi    + 1e-16) - 0.5 * d2_1 / (temperature * sigma2)
        log_r2 = np.log(1.-pi + 1e-16) - 0.5 * d2_2 / (temperature * sigma2)
        both = np.concatenate([log_r1, log_r2], axis=1)
        both -= both.max(axis=1, keepdims=True)
        both = np.exp(both)
        both /= both.sum(axis=1, keepdims=True)
        R1 = both[:, :k]
        R2 = both[:, k:]
        B1 = R1.T @ data
        B2 = R2.T @ data
        counts1 = R1.sum(axis=0)
        counts2 = R2.sum(axis=0)

        # M-step for pi
        pi = R1.sum()/(R1.sum()+R2.sum())

        # M-step for first image
        for _ in range(n_inner):
            pred1 = radon_rows(x1, candidate_angles)
            grad1 = np.zeros_like(x1)

            # Gradient ...
            for m, ang in enumerate(candidate_angles):
                if counts1[m] < 1e-12:
                    continue
                resid1 = counts1[m] * pred1[m] - B1[m]
                grad1 += backproject_single(resid1, ang, d)

            # ... decend
            grad1 = grad1 / sigma2 + lam * x1
            x1 = x1 - lr * grad1
            x1 = np.clip(x1, 0.0, None)
            x1 = np.where(mask, x1, 0.0)

        # M-step for second image
        for _ in range(n_inner):
            pred2 = radon_rows(x2, candidate_angles)
            grad2 = np.zeros_like(x2)

            # Gradient ...
            for m, ang in enumerate(candidate_angles):
                if counts2[m] < 1e-12:
                    continue
                resid2 = counts2[m] * pred2[m] - B2[m]
                grad2 += backproject_single(resid2, ang, d)

            # ... decend
            grad2 = grad2 / sigma2 + lam * x2
            x2 = x2 - lr * grad2
            x2 = np.clip(x2, 0.0, None)
            x2 = np.where(mask, x2, 0.0)

        # Storate
        x1s.append(x1.copy())
        x2s.append(x2.copy())
        pis.append(pi)

        # Showing progress
        if verbose >= 0:
            if ((em_it + 1) % verbose == 0) or (em_it == 0):
                title=f"EM iter {em_it+1:04d} | "+\
                      f"temp={temperature:.4f} | "+\
                      f"variance={sigma2:.4f} | "+\
                      f"pi={pi:.4f}"
                fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.7))
                axes[0].imshow(x1s[-1], cmap='gray')
                axes[0].set_title("Image 1")
                axes[0].set_xticks([])
                axes[0].set_yticks([])
                axes[1].imshow(x2s[-1], cmap='gray')
                axes[1].set_title("Image 2")
                axes[1].set_xticks([])
                axes[1].set_yticks([])
                axes[2].bar(plot_x, counts1, width=1.0, linewidth=0)
                axes[2].set_title("Responsibilities 1")
                axes[2].set_xticks([])
                axes[3].bar(plot_x, counts2, width=1.0, linewidth=0)
                axes[3].set_title("Responsibilities 2")
                axes[3].set_xticks([])
                plt.suptitle(title)
                plt.tight_layout()
                if imsave:
                    plt.savefig(f'{dir}/em2ClsIter{em_it}.png')
                if imshow: plt.show()
                plt.close()

    return x1s, x2s, pis
