import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import PCA
from scipy.ndimage import rotate

from src.em import em_algorithm
from src.gibbs import vanilla_gibbs_sampler, gibbs_LD_sampler, pack_image
from src.metrics import measure_projection_error, \
                        measure_projection_error_minimal, \
                        measure_alignment_error, \
                        measure_projection_error_batch, \
                        measure_alignment_error_batch

# pca embed function
def pca_embed_images(images, n_components=2, normalize=True):
    """
    images: list or array of shape (n, d, d)
    returns: array of shape (n, n_components)
    """
    X = np.asarray(images)
    n, d, _ = X.shape

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

    # Reshape components back to image shape
    components = pca.components_.reshape(n_components, d, d)
    mean_image = pca.mean_.reshape(d, d)

    return X_embedded, pca.explained_variance_ratio_, components, mean_image

# Default parameters
_em_params1_df = {
    'n_em': 200, 'n_inner': 50, 'lr':1e-4, 'lam':5e-3,
    'temp_start':2.0, 'temp_end': 1.0, 'temp_decay':0.995,
    'sigma2':None, 'verbose':10, 'x_init':None
}

_em_params2_df = {
    'n_em': 300, 'n_inner': 50, 'lr':1e-5, 'lam':5e-3,
    'temp_start':2.0, 'temp_end': 1.0, 'temp_decay':0.995,
    'verbose':10
}

_vg_params_df = {
    'n_samples': 1200, 'n_burnins': 0, 'verbose': 10, 'sigma2': 1.
}

_gld_params_df = {
    'n_gibbs': 1200, 'n_burnins': 0, 'lr':1e-5, 'lam':5e-2,
    'temp_start':1.0, 'temp_end': 1.0, 'temp_decay':1.,
    'verbose':10
}

def MCMC_experiment(d, img, data, seed, candidate_angles, sigma2,
                    em_params1=dict(), em_params2=dict(), vg_params=dict(), gld_params=dict(),
                    save_dir='MCMCexp'):
    
    # Directories
    dirs = [f'{save_dir}/em_raw', f'{save_dir}/em_refined', 
            f'{save_dir}/vg', f'{save_dir}/gld', 
            f'{save_dir}/results']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    
    # Parameters
    for k, v in _em_params1_df.items():
        if k not in em_params1: em_params1[k] = v
    for k, v in _em_params2_df.items():
        if k not in em_params2: em_params2[k] = v
    for k, v in _vg_params_df.items():
        if k not in vg_params: vg_params[k] = v
    for k, v in _gld_params_df.items():
        if k not in gld_params: gld_params[k] = v

    # Run EM and MCMC 
    xs_em = em_algorithm(data, candidate_angles, 
                         imshow=False, imsave=True, dir=f'{save_dir}/em_raw',
                         seed=seed, **em_params1)
    
    xs_em_refined = em_algorithm(data, candidate_angles, sigma2=sigma2, x_init=xs_em[-1],
                                 imshow=False, imsave=True, dir=f'{save_dir}/em_refined', 
                                 seed=seed, **em_params2)
    
    vg_samples_xs, vg_samples_zs = vanilla_gibbs_sampler(data, candidate_angles, sigma_eps2=sigma2,
                                                         dir=f'{save_dir}/vg',
                                                         imshow=False, imsave=True,
                                                         x_init=pack_image(xs_em_refined[-1], d),
                                                         random_state=seed, **vg_params)
    
    ld_samples_xs, ld_samples_zs = gibbs_LD_sampler(data, candidate_angles,
                                                    sigma2=sigma2, x_init=xs_em_refined[-1],
                                                    imshow=False, imsave=True, dir=f'{save_dir}/gld',
                                                    seed=seed, **gld_params)
    
    # Report errors for em
    print('Finish simulating! Visualizing output ...')
    print('Calculating EM errors ...')
    proj_error_em = measure_projection_error_minimal(xs_em_refined[-1], data, candidate_angles)
    algm_error_em = measure_alignment_error(xs_em_refined[-1], img, candidate_angles)
    with open(f'{save_dir}/results/errors_em.txt', 'w') as file:
        file.write(f"Proj. error: {float(proj_error_em):.4f}; algm. error: {float(algm_error_em):.4f}")

    # Calculate errors for vg
    print('Calculating VG error ...')
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
    plt.savefig(f'{save_dir}/results/errors_vg.png')
    plt.close()

    # Calculate errors for GLD
    print('Calculating GLD error ...')
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
    plt.savefig(f'{save_dir}/results/errors_gld.png')
    plt.close()

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

    # Plot images
    print('Plotting last samples ...')
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
    plt.savefig(f'{save_dir}/results/plot_samples.png')
    plt.close()

    # Spin image
    img_spin = img.copy()
    dist = np.inf
    for ang in candidate_angles:
        img_temp = rotate(img, angle=ang, reshape=False)
        dist_temp = np.linalg.norm(img_temp-xs_em_refined[-1])
        if dist_temp < dist:
            dist = dist_temp
            img_spin = img_temp

    # pca plot prep
    print('Plotting pca ...')
    n_vg = vg_params['n_samples']
    n_gld = gld_params['n_gibbs']
    classes = np.array([0, 1] + 
                       [2 for _ in range(n_vg//4)] + 
                       [3 for _ in range((n_vg+1)//4)] + 
                       [4 for _ in range((n_vg+2)//4)] + 
                       [5 for _ in range((n_vg+3)//4)] + 
                       [6 for _ in range(n_gld//4)] + 
                       [7 for _ in range((n_gld+1)//4)] + 
                       [8 for _ in range((n_gld+2)//4)] + 
                       [9 for _ in range((n_gld+3)//4)])
    imgs_all = [img_spin, xs_em_refined[-1]] + vg_samples_xs + ld_samples_xs
    imgs_all_encode, pca_ratios, pca_components, pca_mean = pca_embed_images(imgs_all, n_components=2, normalize=False)

    # Plotting
    names = ['True', 'EM', 'VG1', 'VG1', 'VG1', 'VG1',
             'LDG1', 'LDG2', 'LDG3', 'LDG4']
    colors = ['C1', 'C3', '#2CA02C', '#259025', '#1A6E1A', '#0F4D0F',
              '#ADD7F6', '#87BFFF', '#3F8EFC', '#2667FF']
    for i, (name, color) in enumerate(zip(names, colors)):
        taken = (classes == i)
        plt.scatter(imgs_all_encode[taken][:, 0], imgs_all_encode[taken][:, 1], c=color, label=name, s=10, alpha=0.7, zorder=10-i)
    plt.title('PCA of true image and generated estimates - S shaped')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'{save_dir}/results/plot_pca.png', bbox_inches='tight')
    plt.close()

    # Plotting pca components
    print('Plotting pca components ...')
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.7))
    c_min = min(np.min(pca_components), np.min(pca_mean))
    c_max = max(np.max(pca_components), np.max(pca_mean))
    axes[0].imshow(pca_components[0], cmap='gray', vmin=c_min, vmax=c_max)
    axes[1].imshow(pca_components[1], cmap='gray', vmin=c_min, vmax=c_max)
    axes[2].imshow(pca_mean, cmap='gray', vmin=c_min, vmax=c_max)
    axes[0].set_title(f'Var explained: {pca_ratios[0]:.4f}')
    axes[1].set_title(f'Var explained: {pca_ratios[1]:.4f}')
    axes[2].set_title(f'PCA center')
    plt.suptitle('PCA\'s first 2 components')
    plt.savefig(f'{save_dir}/results/plot_pcaComps.png', bbox_inches='tight')
    plt.close()
