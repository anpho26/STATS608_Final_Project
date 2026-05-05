import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.special import logsumexp
from tqdm.auto import tqdm
from IPython.display import clear_output, display

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon, iradon, resize

from src.utils import circle_mask, radon_rows, random_init, backproject_single

# Pack and unpack image - for the Gibbs sampler
def pack_image(img, d):
    """
    Extracts pixels inside circle mask into a 1D vector.
    """
    return img[circle_mask(d)]

def unpack_image(vec, d):
    """
    Reconstructs full d x d image from packed vector.
    Outside-circle pixels are set to 0.
    """
    img = np.zeros((d, d), dtype=vec.dtype)
    img[circle_mask(d)] = vec
    return img

# Build random radon transformation matrix - for the Gibbs sampler
def radon_matrix(d, angles):
    """
    Builds linear operator A such that:
        sinogram_vec = A @ image_packed

    Returns:
        A: shape (d * len(angles), num_circle_pixels)
        mask: circle mask used for packing
    """
    mask = circle_mask(d)
    idx = np.argwhere(mask)
    p = idx.shape[0]
    k = len(angles)
    A = np.zeros((d * k, p))

    for j in range(p):
        basis = np.zeros((d, d))
        y, x = idx[j]
        basis[y, x] = 1.0
        sino = radon(basis, theta=angles, circle=True, preserve_range=True).T
        A[:, j] = sino.reshape(-1)

    return A

# Sample from a linear Gaussian model, with the help of AI
def sample_posterior_A(Y, B, sigma2, diag_sigma_eps, n_samples=1, random_state=None):
    """
    Sample from posterior of A in linear Gaussian model:
        A ~ N(0, sigma2 I)
        Y = B A + eps, eps ~ N(0, diag(diag_sigma_eps))

    Parameters
    ----------
    Y : (m,) array
    B : (m, n) array
    sigma2 : float
    diag_sigma_eps : (m,) array  (diagonal entries of Sigma_eps)
    n_samples : int
    random_state : int or None

    Returns
    -------
    samples : (n_samples, n) array
    """

    rng = np.random.default_rng(random_state)
    m, n = B.shape

    # Inverse of diagonal noise covariance
    Sigma_eps_inv = 1.0 / diag_sigma_eps  # shape (m,)

    # Compute precision matrix: Lambda = (1/sigma2) I + B^T Sigma_eps^{-1} B
    # Efficiently: weight rows of B by sqrt(inv variance)
    W = Sigma_eps_inv  # (m,)
    BW = B.T * W  # (n, m)

    Lambda = (1.0 / sigma2) * np.eye(n) + BW @ B  # (n, n)

    # Cholesky of precision (more stable than covariance inversion)
    L = np.linalg.cholesky(Lambda)  # Lambda = L L^T

    # Compute mean: mu = Lambda^{-1} B^T Sigma_eps^{-1} Y
    rhs = B.T @ (Sigma_eps_inv * Y)

    # Solve Lambda mu = rhs using Cholesky
    # First solve L z = rhs, then L^T mu = z
    z = np.linalg.solve(L, rhs)
    mu = np.linalg.solve(L.T, z)

    # Sampling: A = mu + Lambda^{-1/2} * z
    # To sample, solve L^T x = normal(0, I)
    samples = []
    for _ in range(n_samples):
        eps = rng.standard_normal(n)
        # Solve L^T x = eps  => x = L^{-T} eps
        x = np.linalg.solve(L.T, eps)
        samples.append(mu + x)

    return np.array(samples)

# Helper function to calculate group count and group means, with help from AI
def group_means(A, B1, B2, atol=1e-8, rtol=1e-5):
    """
    Parameters
    ----------
    A : (n,) array of unique-ish float values
    B1 : (m,) array of float values (repeats from A)
    B2 : (m, k) array
    atol, rtol : tolerances for np.isclose

    Returns
    -------
    C : (n, k) array
        Row i = mean of B2 rows where B1 is close to A[i]
    D : (n,) array
        Counts of matches
    """

    A = np.asarray(A)
    B1 = np.asarray(B1)
    B2 = np.asarray(B2)
    n = len(A)
    k = B2.shape[1]
    C = np.zeros((n, k))
    D = np.zeros(n, dtype=int)

    for i, a in enumerate(A):
        mask = np.isclose(B1, a, atol=atol, rtol=rtol)
        D[i] = np.sum(mask)
        if D[i] > 0: C[i] = B2[mask].mean(axis=0)
        else: C[i] = np.nan

    return C, D

# Simple "Gibbs samplar" when the angles are known (technically not a Gibbs sampler since there is no latent).
def gibbs_sample_radon_known_angles(data, proj_angles, candidate_angles, sigma2, sigma_eps2, n_samples=1, \
                                    random_state=None, radon_map=None, pack=False):

    # Format observed data
    n, d = data.shape
    k = len(candidate_angles)
    if radon_map is None: radon_map = radon_matrix(d, candidate_angles)
    angle_means, angle_counts = group_means(candidate_angles, proj_angles, data)
    obs_bool = angle_counts > 0
    obs_means = angle_means[obs_bool]
    obs_counts = angle_counts[obs_bool]
    obs_map = radon_map[np.repeat(obs_bool, d), :]

    # Format arguments for the linear Gaussian setup
    obs_angles_flatten = obs_means.reshape(-1)
    obs_variance = np.repeat(sigma_eps2/obs_counts, d)

    # Use linear Gaussian posterior formula and return
    samples = sample_posterior_A(obs_angles_flatten, obs_map, sigma2, obs_variance, n_samples=n_samples, random_state=random_state)
    return np.array(samples) if pack else np.array([unpack_image(vec, d) for vec in samples])

# Vanilla Gibbs sampler
def vanilla_gibbs_sampler(data, candidate_angles, sigma2, sigma_eps2, n_samples=1, n_burnins=0,
                          random_state=None, x_init=None, verbose=0,
                          imshow=True, imsave=False, dir='plotsVG'):

    # Setups
    np.random.seed(random_state)
    n, d = data.shape
    k = len(candidate_angles)
    mask_d = circle_mask(d)
    p = mask_d.sum()
    radon_map = radon_matrix(d, candidate_angles)
    os.makedirs(dir, exist_ok=True)

    # Init & storage
    x_sum = np.zeros((d, d))
    x = np.random.standard_normal(p)*np.sqrt(sigma2) if x_init is None else x_init.copy()
    if len(x.shape) == 2: x = pack_image(x, d)
    z = np.random.randint(k, size=n)
    samples_x = []
    samples_z = []
    # print(x.shape, z.shape, mask_d.shape, radon_map.shape)

    # Main loop
    for it in tqdm(range(n_samples+n_burnins)):

        # Sample z | x, y (data)
        sino = (x @ radon_map.T).reshape((k, d))
        counts = []
        for i in range(n):
            resid = data[i]-sino
            logp = -0.5 / sigma_eps2 * np.sum(resid*resid, axis=1)
            logp -= logsumexp(logp)
            temp_ind = np.random.choice(k, p=np.exp(logp))
            counts.append(temp_ind)
            z[i] = candidate_angles[temp_ind]

        # Sample x | z, y
        x = gibbs_sample_radon_known_angles(data, z, candidate_angles, sigma2=sigma2,
                                            sigma_eps2=sigma_eps2, radon_map=radon_map)[0]

        # Store
        if it >= n_burnins: x_sum += x
        samples_x.append(x.copy())
        samples_z.append(z.copy())
        x = pack_image(x, d)

        # Verbose
        if verbose > 0 and it % verbose == 0:
            if it >= n_burnins:
                fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.7))
                axes[0].imshow(samples_x[-1], cmap='gray')
                axes[0].set_title("Current sample")
                axes[0].set_xticks([])
                axes[0].set_yticks([])
                axes[1].imshow(x_sum/(it+1-n_burnins), cmap='gray')
                axes[1].set_title("Running mean")
                axes[1].set_xticks([])
                axes[1].set_yticks([])
                axes[2].hist(counts, bins=np.arange(k+1)-0.5)
                axes[2].set_xticks([])
                plt.suptitle(f"Plottings at iteration {it}:")
                plt.tight_layout()
                if imsave:
                    plt.savefig(f'{dir}/VGiter{it}.png')
                if imshow: plt.show()
                plt.close()
            else:
                fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
                axes[0].imshow(samples_x[-1], cmap='gray')
                axes[0].set_title("Current sample")
                axes[0].set_xticks([])
                axes[0].set_yticks([])
                axes[1].hist(counts, bins=np.arange(k+1)-0.5)
                axes[1].set_xticks([])
                plt.suptitle(f"Plottings at iteration {it}:")
                plt.tight_layout()
                if imsave:
                    plt.savefig(f'{dir}/VGiter{it}.png')
                if imshow: plt.show()
                plt.close()

    # Return
    return samples_x[n_burnins:], samples_z[n_burnins:]

# Gibbs - Langevin Dynamics sampler
def gibbs_LD_sampler(data, candidate_angles,
                     n_gibbs=100, n_burnins=0, n_inner=50, lr=1e-4, lam=5e-3,
                     temp_start=2.0, temp_end=1.0, temp_decay=0.995,
                     seed=0, sigma2=None, verbose=-1, x_init=None,
                     imshow=True, imsave=False, dir='plotsGLD'):

    # Set ups
    n, d = data.shape
    k = len(candidate_angles)
    mask = circle_mask(d)
    plot_x = list(range(k))
    if sigma2 is None: sigma2 = np.var(data)
    pbar = tqdm(range(n_gibbs+n_burnins), desc="Gibbs LD iterations")
    os.makedirs(dir, exist_ok=True)

    # Initialization
    x = random_init(d, seed=seed) if x_init is None else x_init.copy()
    x = np.where(mask, x, 0.0)
    z = np.random.randint(k, size=n)
    xs = [x.copy()]
    zs = [z.copy()]
    x_mean = np.zeros_like(x)

    # Main loop
    for it in pbar:
        temperature = max(temp_end, temp_start * (temp_decay ** it))

        # Sample z | x, y (data)
        pred = radon_rows(x, candidate_angles)
        for i in range(n):
            resid = data[i]-pred
            logp = -0.5 * np.sum(resid*resid, axis=1) / (temperature * sigma2)
            logp -= logsumexp(logp)
            z[i] = np.random.choice(np.arange(k), p=np.exp(logp))

        R = np.zeros((n, k), dtype='int')
        R[np.arange(n), z] = 1
        B = R.T @ data
        counts = R.sum(axis=0)

        # Sample  x | z, y
        for _ in range(n_inner):
            pred = radon_rows(x, candidate_angles)
            grad = np.zeros_like(x)

            # Gradient calculation
            for m, ang in enumerate(candidate_angles):
                if counts[m] < 1e-12:
                    continue
                resid = counts[m] * pred[m] - B[m]
                grad += backproject_single(resid, ang, d)

            # Langevin Dynamics
            grad = grad / sigma2 + lam * x
            x = x - lr * grad + np.sqrt(2 * lr) * np.random.randn(*x.shape)
            x = np.where(mask, x, 0.0)

        # Storate
        xs.append(x.copy())
        zs.append(z.copy())
        if it >= n_burnins: x_mean += x.copy()

        # Showing progress
        if verbose >= 0:
            if ((it + 1) % verbose == 0) or (it == 0):
                if it >= n_burnins:
                    title=f"Gibbs LD iter {it+1:04d} | "+\
                          f"temp={temperature:.4f} | "+\
                          f"variance={sigma2:.4f} "
                    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.7))
                    axes[0].imshow(xs[-1], cmap='gray')
                    axes[0].set_title("Current sample")
                    axes[0].set_xticks([])
                    axes[0].set_yticks([])
                    axes[1].imshow(x_mean/(it+1-n_burnins), cmap='gray')
                    axes[1].set_title("Running mean")
                    axes[1].set_xticks([])
                    axes[1].set_yticks([])
                    axes[2].bar(plot_x, counts, width=1.0, linewidth=0)
                    axes[2].set_title(f"Label count")
                    axes[2].set_xticks([])
                    plt.suptitle(title)
                    plt.tight_layout()
                    if imsave:
                        plt.savefig(f'{dir}/GLDiter{it}.png')
                    if imshow: plt.show()
                    plt.close()
                else:
                    title=f"Gibbs LD iter {it+1:04d} | "+\
                          f"temp={temperature:.4f} | "+\
                          f"variance={sigma2:.4f} "
                    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
                    axes[0].imshow(xs[-1], cmap='gray')
                    axes[0].set_title("Current sample")
                    axes[0].set_xticks([])
                    axes[0].set_yticks([])
                    axes[1].bar(plot_x, counts, width=1.0, linewidth=0)
                    axes[1].set_title(f"Label count")
                    axes[1].set_xticks([])
                    plt.suptitle(title)
                    plt.tight_layout()
                    if imsave:
                        plt.savefig(f'{dir}/GLDiter{it}.png')
                    if imshow: plt.show()
                    plt.close()


    return xs[n_burnins+1:], zs[n_burnins+1:]