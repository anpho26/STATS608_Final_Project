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
                                  random_state=None, x_init=None, verbose=0):

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
            print(f"Plottings at iteration {it}:")
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
            plt.tight_layout()
            plt.show()

    # Return
    return samples_x[n_burnins:], samples_z[n_burnins:], samples_c[n_burnins:], samples_pi[n_burnins:]