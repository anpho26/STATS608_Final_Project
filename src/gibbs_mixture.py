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
from src.gibbs import pack_image, unpack_image, radon_matrix, gibbs_sample_radon_known_angles

# Vanilla Gibbs sampler
def vanilla_gibbs_mixture_sampler(data, candidate_angles, sigma2, sigma_eps2, alpha=1., n_mixtures=1, n_samples=1, n_burnins=0,
                                  random_state=None, x_init=None, verbose=0,
                                  imshow=True, imsave=False, dir='plotsGibbsMix'):

    # Setups
    np.random.seed(random_state)
    n, d = data.shape
    k = len(candidate_angles)
    m = n_mixtures
    mask_d = circle_mask(d)
    p = mask_d.sum()
    radon_map = radon_matrix(d, candidate_angles)
    alpha = np.array(alpha)
    if len(alpha) == 1: alpha = np.array([alpha[0] for _ in range(m)])
    os.makedirs(dir, exist_ok=True)

    # Init & storage
    x_sum = np.zeros((m, d, d))
    x = np.random.standard_normal((m, p))*np.sqrt(sigma2) if x_init is None else x_init.copy()
    if len(x.shape) == 3: x = np.array([pack_image(img, d) for img in x_init])
    z = np.random.randint(k, size=n)
    c = np.random.randint(m, size=n)
    pi = np.random.dirichlet(alpha)
    samples_x = []
    samples_z = []
    samples_c = []
    samples_pi = []
    # print(x.shape, z.shape, mask_d.shape, radon_map.shape)

    # Main loop
    for it in tqdm(range(n_samples+n_burnins)):

        # Sample z | x, pi, y (data)
        sino = (x @ radon_map.T).reshape((m, k, d))
        for i in range(n):
            resid = data[i]-sino
            logp = -0.5 / sigma_eps2 * np.sum(resid*resid, axis=-1)
            logp += np.log(pi)[:, None]
            logp -= logsumexp(logp)
            p_flat = np.exp(logp).ravel()
            idx = np.random.choice(p_flat.size, p=p_flat)
            m1, k1 = divmod(idx, k)
            z[i] = candidate_angles[k1]
            c[i] = m1

        # Sample pi | x, y, z
        pi = np.random.dirichlet(alpha+np.bincount(c, minlength=m))

        # Sample x | z, pi, y
        for i in range(m):
            flag = c == i
            x[i] = gibbs_sample_radon_known_angles(data[flag], z[flag], candidate_angles, sigma2=sigma2,
                                                   sigma_eps2=sigma_eps2, radon_map=radon_map, pack=True)[0]

        # Store
        x_image = np.array([unpack_image(x[i], d) for i in range(m)])
        if it >= n_burnins: x_sum += x_image
        samples_x.append(x_image)
        samples_z.append(z.copy())
        samples_c.append(c.copy())
        samples_pi.append(pi.copy())

        # Verbose
        if verbose > 0 and it % verbose == 0:
            fig, axes = plt.subplots(2, (n_mixtures+1), figsize=(2.7*(n_mixtures+1), 5.4))
            for i in range(m):
                axes[0][i].imshow(samples_x[-1][i], cmap='gray')
                axes[0][i].set_title(f"Current {i}-th")
                axes[0][i].set_xticks([])
                axes[0][i].set_yticks([])
                if it >= n_burnins:
                    axes[1][i].imshow(x_sum[i]/(it+1-n_burnins), cmap='gray')
                    axes[1][i].set_title("Running mean")
                    axes[1][i].set_xticks([])
                    axes[1][i].set_yticks([])
                else: axes[1][i].axis('off')
            axes[0][m].hist(z, bins=np.arange(k+1) - 0.5)
            axes[0][m].set_xticks([])
            axes[0][m].set_title("Angle dist.")
            axes[1][m].hist(c, bins=np.arange(m+1) - 0.5)
            axes[1][m].set_xticks([])
            axes[1][m].set_title("Class dist.")
            plt.suptitle(f"Plottings at iteration {it}:")
            plt.tight_layout()
            if imsave:
                plt.savefig(f'{dir}/GibbsMixIter{it}.png')
            if imshow: plt.show()
            plt.close()

    # Return
    return samples_x[n_burnins:], samples_z[n_burnins:], samples_c[n_burnins:], samples_pi[n_burnins:]

# Some helpers for MALA mixture

def compute_grad_U_mixture_component(
    x_j,
    data_j,
    z_j,
    candidate_angles,
    sigma2,
    lam,
):
    """
    Gradient for one mixture component image x_j,
    conditional on observations assigned to this component.
    """
    d = x_j.shape[0]
    k = len(candidate_angles)

    if len(data_j) == 0:
        return lam * x_j

    pred = radon_rows(x_j, candidate_angles)

    R = np.zeros((len(z_j), k))
    R[np.arange(len(z_j)), z_j] = 1

    B = R.T @ data_j
    counts = R.sum(axis=0)

    grad = np.zeros_like(x_j)

    for m, ang in enumerate(candidate_angles):
        if counts[m] < 1e-12:
            continue

        resid = counts[m] * pred[m] - B[m]
        grad += backproject_single(resid, ang, d)

    return grad / sigma2 + lam * x_j

def compute_U_mixture_component(
    x_j,
    data_j,
    z_j,
    candidate_angles,
    sigma2,
    lam,
):
    """
    Energy for one mixture component image x_j,
    conditional on observations assigned to this component.
    """
    if len(data_j) == 0:
        return 0.5 * lam * np.sum(x_j ** 2)

    pred = radon_rows(x_j, candidate_angles)
    resid = data_j - pred[z_j]

    data_term = 0.5 / sigma2 * np.sum(resid ** 2)
    prior_term = 0.5 * lam * np.sum(x_j ** 2)

    return data_term + prior_term

def log_q(x_from, x_to, grad_from, step):
    mean = x_from - step * grad_from
    diff = x_to - mean
    return -np.sum(diff ** 2) / (4 * step)

def mala_step_mixture_component(
    x_j,
    data_j,
    z_j,
    candidate_angles,
    sigma2,
    lam,
    step,
    mask,
):
    grad_x = compute_grad_U_mixture_component(
        x_j,
        data_j,
        z_j,
        candidate_angles,
        sigma2,
        lam,
    )

    noise = np.random.randn(*x_j.shape)
    x_prop = x_j - step * grad_x + np.sqrt(2 * step) * noise
    x_prop = np.where(mask, x_prop, 0.0)

    U_x = compute_U_mixture_component(
        x_j,
        data_j,
        z_j,
        candidate_angles,
        sigma2,
        lam,
    )

    U_prop = compute_U_mixture_component(
        x_prop,
        data_j,
        z_j,
        candidate_angles,
        sigma2,
        lam,
    )

    grad_prop = compute_grad_U_mixture_component(
        x_prop,
        data_j,
        z_j,
        candidate_angles,
        sigma2,
        lam,
    )

    log_forward = log_q(x_j, x_prop, grad_x, step)
    log_backward = log_q(x_prop, x_j, grad_prop, step)

    log_alpha = -U_prop + U_x + log_backward - log_forward

    if np.log(np.random.rand()) < log_alpha:
        return x_prop, True
    else:
        return x_j, False


def MALA_mixture_sampler(
    data,
    candidate_angles,
    sigma2,
    alpha=1.0,
    n_mixtures=2,
    n_gibbs=100,
    n_burnins=0,
    n_inner=50,
    lr=1e-4,
    lam=5e-3,
    temp_start=2.0,
    temp_end=1.0,
    temp_decay=0.995,
    seed=0,
    x_init=None,
    verbose=-1,
    imshow=True,
    imsave=False,
    dir="plotsMALAMix",
):
    """
    Gibbs-within-MALA sampler for a mixture of images.

    Latents:
        c_i = mixture component
        z_i = angle index

    Parameters
    ----------
    data : array, shape (n, d)
        Observed projections.
    candidate_angles : array, shape (k,)
        Discrete angle grid used for inference.
    sigma2 : float
        Observation noise variance.
    alpha : float or array
        Dirichlet prior parameter for mixture weights.
    n_mixtures : int
        Number of mixture components. For your case, use 2.
    """

    np.random.seed(seed)

    n, d = data.shape
    k = len(candidate_angles)
    m = n_mixtures
    mask = circle_mask(d)
    os.makedirs(dir, exist_ok=True)

    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim == 0:
        alpha = np.repeat(alpha, m)
    if len(alpha) == 1:
        alpha = np.repeat(alpha[0], m)

    # Initialize images
    if x_init is None:
        x = np.array([
            random_init(d, seed=seed + j)
            for j in range(m)
        ])
    else:
        x = x_init.copy()

    for j in range(m):
        x[j] = np.where(mask, x[j], 0.0)

    # Initialize latent variables
    z = np.random.randint(k, size=n)
    c = np.random.randint(m, size=n)
    pi = np.random.dirichlet(alpha)

    xs = []
    zs = []
    cs = []
    pis = []

    x_mean = np.zeros_like(x)

    pbar = tqdm(range(n_gibbs + n_burnins), desc="Mixture MALA iterations")

    for it in pbar:
        temperature = max(temp_end, temp_start * (temp_decay ** it))

        # ------------------------------------------------------------
        # 1. Sample (c_i, z_i) | x, pi, y_i
        # ------------------------------------------------------------
        pred = np.array([
            radon_rows(x[j], candidate_angles)
            for j in range(m)
        ])
        # pred shape: (m, k, d)

        for i in range(n):
            resid = data[i][None, None, :] - pred
            logp = -0.5 * np.sum(resid ** 2, axis=-1) / (temperature * sigma2)
            logp += np.log(pi)[:, None]

            logp -= logsumexp(logp)
            p_flat = np.exp(logp).ravel()

            idx = np.random.choice(p_flat.size, p=p_flat)
            c_i, z_i = divmod(idx, k)

            c[i] = c_i
            z[i] = z_i

        # ------------------------------------------------------------
        # 2. Sample pi | c
        # ------------------------------------------------------------
        counts_c = np.bincount(c, minlength=m)
        pi = np.random.dirichlet(alpha + counts_c)

        # ------------------------------------------------------------
        # 3. Sample each image x_j | c, z, y using MALA
        # ------------------------------------------------------------
        accepts = np.zeros(m)

        for j in range(m):
            idx_j = c == j
            data_j = data[idx_j]
            z_j = z[idx_j]

            for _ in range(n_inner):
                x[j], accepted = mala_step_mixture_component(
                    x[j],
                    data_j,
                    z_j,
                    candidate_angles,
                    sigma2,
                    lam,
                    lr,
                    mask,
                )
                accepts[j] += accepted

        acc_rates = accepts / n_inner

        # ------------------------------------------------------------
        # Store
        # ------------------------------------------------------------
        xs.append(x.copy())
        zs.append(z.copy())
        cs.append(c.copy())
        pis.append(pi.copy())

        if it >= n_burnins:
            x_mean += x.copy()

        pbar.set_postfix({
            "temp": f"{temperature:.3f}",
            "acc": ",".join([f"{a:.2f}" for a in acc_rates]),
            "pi": ",".join([f"{p:.2f}" for p in pi]),
        })

        # ------------------------------------------------------------
        # Plot progress
        # ------------------------------------------------------------
        if verbose >= 0:
            if ((it + 1) % verbose == 0) or (it == 0):

                fig, axes = plt.subplots(
                    2,
                    m + 1,
                    figsize=(2.7 * (m + 1), 5.4),
                )

                for j in range(m):
                    axes[0, j].imshow(x[j], cmap="gray")
                    axes[0, j].set_title(f"Current comp {j}")
                    axes[0, j].set_xticks([])
                    axes[0, j].set_yticks([])

                    if it >= n_burnins:
                        denom = it + 1 - n_burnins
                        axes[1, j].imshow(x_mean[j] / denom, cmap="gray")
                        axes[1, j].set_title(f"Mean comp {j}")
                        axes[1, j].set_xticks([])
                        axes[1, j].set_yticks([])
                    else:
                        axes[1, j].axis("off")

                axes[0, m].hist(z, bins=np.arange(k + 1) - 0.5)
                axes[0, m].set_title("Angle labels")
                axes[0, m].set_xticks([])

                axes[1, m].hist(c, bins=np.arange(m + 1) - 0.5)
                axes[1, m].set_title("Class labels")
                axes[1, m].set_xticks([])

                title = (
                    f"Mixture MALA iter {it+1:04d} | "
                    f"temp={temperature:.4f} | "
                    f"pi={np.round(pi, 3)} | "
                    f"acc={np.round(acc_rates, 3)}"
                )

                plt.suptitle(title)
                plt.tight_layout()

                if imsave:
                    plt.savefig(f"{dir}/MALAMixIter{it}.png")

                if imshow:
                    plt.show()

                plt.close()

    return (
        xs[n_burnins:],
        zs[n_burnins:],
        cs[n_burnins:],
        pis[n_burnins:],
    )