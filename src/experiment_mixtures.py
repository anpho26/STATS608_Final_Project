import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import PCA
from scipy.ndimage import rotate

from src.utils import (
    load_image,
    circle_mask,
    simulate_mixture_data,
    make_symbol,
)

from src.gibbs_mixture import (
    vanilla_gibbs_mixture_sampler,
    MALA_mixture_sampler,
)

from src.em import (
    em_algorithm,
    em_algorithm_2classes,
)

# Helper plotting function
def plot_column(ax0, ax1, ax2, a, b, x, y, bins=30):
    # common x-range
    xmin = min(a, b, np.min(x), np.min(y))
    xmax = max(a, b, np.max(x), np.max(y))

    # make the bars thin relative to the range
    width = 0.02 * (xmax - xmin)
    if width == 0:
        width = 1.0

    # first subplot: a (ground truth) and b
    h = max(
        np.histogram(x, bins=bins)[0].max(),
        np.histogram(y, bins=bins)[0].max())
    ax0.bar(a, h, width=width, color='tab:red')
    ax0.bar(b, h, width=width)  # default color
    ax0.set_title('1')

    # second and third subplots
    ax1.hist(x, bins=bins)
    ax1.set_title('2')
    ax2.hist(y, bins=bins)
    ax2.set_title('3')

    # enforce common x-axis
    for ax in (ax0, ax1, ax2):
        ax.set_xlim(xmin, xmax)

    return xmin, xmax

def MCMC_experiment_mixture2(d, img1, img2, pi0, 
                             data, candidate_angles,
                             seed, sigma2, sigma_eps2,
                             em_params1=dict(), em_params2=dict(),
                             vg_params = dict(), mala_params=dict(),
                             save_dir='MCMCexp_mix'):
    
    # Directories
    dirs = [f'{save_dir}/em_raw', f'{save_dir}/em_refined', 
            f'{save_dir}/vg', f'{save_dir}/mala',
            f'{save_dir}/results']
    
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    # Parameters
    # for k, v in _em_params1_df.items():
    #     if k not in em_params1: em_params1[k] = v
    # for k, v in _em_params2_df.items():
    #     if k not in em_params2: em_params2[k] = v
    # for k, v in _vg_params_df.items():
    #     if k not in vg_params: vg_params[k] = v
    # for k, v in _mala_params_df.items():
    #     if k not in mala_params:
    #         mala_params[k] = v

    # Run EM mixture
    print(f"\nRunning EM mixture, seed={seed}")
    x1s, x2s, pis = em_algorithm_2classes(data, candidate_angles, 
                                          n_em=200, n_inner=50, lr=1e-4, lam=5e-3,
                                          temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                                          seed=seed, sigma2=None, verbose=10,
                                          x_init1=None, x_init2=None, pi_init=0.555,
                                          imshow=False, imsave=True, dir='exp_output/mixture2_em_test')
    
    # Get initialization
    x_init = np.array([x1s[-1], x2s[-1]])

    # Run vanilla Gibbs mixture
    print(f"\nRunning vanilla Gibbs mixture, seed={seed}")
    out_dir =  f'{save_dir}/mala'

    xs_gibbs, zs_gibbs, cs_gibbs, pis_gibbs = vanilla_gibbs_mixture_sampler(
        data=data,
        candidate_angles=candidate_angles,
        sigma2=sigma2,
        sigma_eps2=sigma_eps2,
        alpha=1.0,
        n_mixtures=2,
        n_samples=200,
        n_burnins=50,
        random_state=seed,
        x_init=x_init,
        verbose=10,
        imshow=False,
        imsave=True,
        dir=out_dir,
    )

    np.save(f"{out_dir}/xs_gibbs.npy", np.array(xs_gibbs))
    np.save(f"{out_dir}/zs_gibbs.npy", np.array(zs_gibbs))
    np.save(f"{out_dir}/cs_gibbs.npy", np.array(cs_gibbs))
    np.save(f"{out_dir}/pis_gibbs.npy", np.array(pis_gibbs))
    print(f"Saved Gibbs mixture output to {out_dir}")

    # Run MALA mixture
    print(f"\nRunning MALA mixture, seed={seed}")
    out_dir = f'{save_dir}/vg'
    xs_mala, zs_mala, cs_mala, pis_mala = MALA_mixture_sampler(
        data=data,
        candidate_angles=candidate_angles,
        sigma2=sigma2,
        alpha=1.0,
        n_mixtures=2,
        n_gibbs=200,
        n_burnins=50,
        n_inner=20,
        lr=1e-5,
        lam=5e-3,
        temp_start=2.0,
        temp_end=1.0,
        temp_decay=0.995,
        seed=seed,
        x_init=x_init,
        verbose=10,
        imshow=False,
        imsave=True,
        dir=out_dir,
    )

    np.save(f"{out_dir}/xs_mala.npy", np.array(xs_mala))
    np.save(f"{out_dir}/zs_mala.npy", np.array(zs_mala))
    np.save(f"{out_dir}/cs_mala.npy", np.array(cs_mala))
    np.save(f"{out_dir}/pis_mala.npy", np.array(pis_mala))
    print(f"Saved MALA mixture output to {out_dir}")

    # Compute means
    vg_mean = np.mean(np.array(xs_gibbs), axis=0)
    mala_mean = np.mean(np.array(xs_mala), axis=0)

    # Plot images
    print('Plotting last samples ...')
    fig, axes = plt.subplots(3, (2)*2+1, figsize=((2)*6+5, 10))
    vmin = -1.
    vmax = max(np.max(img1), np.max(img2)) + 1.

    # Row 1: ground truth, EM
    im = axes[0][0].imshow(img1, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0][0].set_title('Ground truth 1')
    axes[0][1].imshow(img2, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0][1].set_title('Ground truth 2')
    axes[0][2].imshow(x1s[-1], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0][2].set_title('EM 1')
    axes[0][3].imshow(x2s[-1], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0][3].set_title('EM 2')

    # Row 2: VG
    axes[1][0].imshow(xs_gibbs[-1][0], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1][0].set_title("VG last 1")
    axes[1][1].imshow(vg_mean[0], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1][1].set_title("VG mean 1")
    axes[1][2].imshow(xs_gibbs[-1][1], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1][2].set_title("VG last 2")
    axes[1][3].imshow(vg_mean[1], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1][3].set_title("VG mean 1")

    # Row 3: MALA
    axes[2][0].imshow(xs_mala[-1][0], cmap='gray', vmin=vmin, vmax=vmax)
    axes[2][0].set_title("MALA last 1")
    axes[2][1].imshow(mala_mean[0], cmap='gray', vmin=vmin, vmax=vmax)
    axes[2][1].set_title("MALA mean 1")
    axes[2][2].imshow(xs_mala[-1][1], cmap='gray', vmin=vmin, vmax=vmax)
    axes[2][2].set_title("MALA last 2")
    axes[2][3].imshow(mala_mean[1], cmap='gray', vmin=vmin, vmax=vmax)
    axes[2][3].set_title("MALA mean 1")

    # Distribution of proportions
    plot_column(axes[0][4], axes[1][4], axes[2][4],
                pi0, pis[-1], np.array(pis_gibbs), np.array(pis_mala))

    # Save image
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    cax = fig.add_axes([0.02, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax)
    plt.savefig(f'{save_dir}/results/plot_samples.png')
    plt.close()
    

