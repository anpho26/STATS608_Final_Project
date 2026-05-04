import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.special import logsumexp
from tqdm.auto import tqdm
from IPython.display import clear_output, display

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon, iradon, resize

# Image loader
def load_image(path, size=32):
    img = imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img.astype(float)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img = resize(img, (size, size), anti_aliasing=True)
    return img

# Circle masking
def circle_mask(d):
    y, x = np.ogrid[:d, :d]
    cy = cx = d / 2.0
    return (x - cx) ** 2 + (y - cy) ** 2 <= ((d-1) // 2) ** 2

# Image generation function
def make_diagonal_rectangle_in_circle(size=32, rect_height=4, rect_width=18, angle=45):
    img = np.zeros((size, size), dtype=float)

    # Centered horizontal rectangle
    r0 = size // 2 - rect_height // 2
    r1 = r0 + rect_height
    c0 = size // 2 - rect_width // 2
    c1 = c0 + rect_width
    img[r0:r1, c0:c1] = 1.0

    # Rotate to make it diagonal
    img = rotate(img, angle=angle, reshape=False, order=1)

    # Keep only the part inside the circle
    mask = circle_mask(size)
    img = np.where(mask, img, 0.0)

    # Normalize to [0,1]
    img = np.clip(img, 0.0, 1.0)
    return img

# Generate an i-shaped image
def generate_i(size=32):
    return make_diagonal_rectangle_in_circle(size, size//8, int(size//1.75), angle=45)

# Generate a square-shaped image
def generate_square(size=32):
    return make_diagonal_rectangle_in_circle(size, int(size//2), int(size//2), angle=0)

# Generate an s-shaped image
def generate_Sshape(size=32, thickness=None, length=None, angle=30):
    if thickness is None: thickness = size//12
    if length is None: length = (size-3)//2
    img = np.zeros((size, size), dtype=float)

    cx, cy = size // 2, size // 2

    # --- Top horizontal bar ---
    r0 = cy - length//2
    r1 = r0 + thickness
    c0 = cx - length//2
    c1 = cx + length//2
    img[r0:r1, c0:c1] = 1.0

    # --- Upper left vertical bar ---
    r0 = cy - length//2
    r1 = cy
    c0 = cx - length//2
    c1 = c0 + thickness
    img[r0:r1, c0:c1] = 1.0

    # --- Middle horizontal bar ---
    r0 = cy - thickness//2
    r1 = r0 + thickness
    c0 = cx - length//2
    c1 = cx + length//2
    img[r0:r1, c0:c1] = 1.0

    # --- Lower right vertical bar ---
    r0 = cy
    r1 = cy + length//2
    c0 = cx + length//2 - thickness
    c1 = cx + length//2
    img[r0:r1, c0:c1] = 1.0

    # --- Bottom horizontal bar ---
    r0 = cy + length//2 - thickness
    r1 = cy + length//2
    c0 = cx - length//2
    c1 = cx + length//2
    img[r0:r1, c0:c1] = 1.0

    # optional rotation (kept for similarity with your original code)
    img = rotate(img, angle=angle, reshape=False, order=1)

    # keep only inside circle
    mask = circle_mask(size)
    img = np.where(mask, img, 0.0)

    # normalize
    img = np.clip(img, 0.0, 1.0)
    return img

# Generate an s-shaped image with random signal
def generate_Sshape_random_signal(
    size=32,
    thickness=None,
    length=None,
    angle=30,
    seed=None,
    low=0.2,
    high=1.0,
):
    """
    Generate an S-shaped image whose support is the same as before,
    but whose pixel intensities on the S-shape are random.

    Parameters
    ----------
    size : int
        Image size.
    thickness : int or None
        Thickness of the bars.
    length : int or None
        Length parameter for the S-shape.
    angle : float
        Rotation angle in degrees.
    seed : int or None
        Random seed.
    low, high : float
        Range of random intensities on the S-shape.

    Returns
    -------
    img : 2D numpy array
        Rotated S-shape with random signal values inside the shape.
    """
    if thickness is None:
        thickness = size // 12
    if length is None:
        length = (size - 3) // 2

    rng = np.random.default_rng(seed)
    support = np.zeros((size, size), dtype=float)

    cx, cy = size // 2, size // 2

    # --- Top horizontal bar ---
    r0 = cy - length // 2
    r1 = r0 + thickness
    c0 = cx - length // 2
    c1 = cx + length // 2
    support[r0:r1, c0:c1] = 1.0

    # --- Upper left vertical bar ---
    r0 = cy - length // 2
    r1 = cy
    c0 = cx - length // 2
    c1 = c0 + thickness
    support[r0:r1, c0:c1] = 1.0

    # --- Middle horizontal bar ---
    r0 = cy - thickness // 2
    r1 = r0 + thickness
    c0 = cx - length // 2
    c1 = cx + length // 2
    support[r0:r1, c0:c1] = 1.0

    # --- Lower right vertical bar ---
    r0 = cy
    r1 = cy + length // 2
    c0 = cx + length // 2 - thickness
    c1 = cx + length // 2
    support[r0:r1, c0:c1] = 1.0

    # --- Bottom horizontal bar ---
    r0 = cy + length // 2 - thickness
    r1 = cy + length // 2
    c0 = cx - length // 2
    c1 = cx + length // 2
    support[r0:r1, c0:c1] = 1.0

    # Fill the support with random values
    img = np.zeros_like(support)
    mask_support = support > 0
    img[mask_support] = rng.uniform(low, high, size=mask_support.sum())

    # Rotate
    img = rotate(img, angle=angle, reshape=False, order=1)

    # Keep only inside circle
    mask = circle_mask(size)
    img = np.where(mask, img, 0.0)

    # Clip to valid range
    img = np.clip(img, 0.0, 1.0)
    return img

# Helper functions for the EM algorithm
def radon_rows(image, angles):
    return radon(image, theta=angles, circle=True, preserve_range=True).T

# Simulate
def simulate_data(image, candidate_angles, n_obs=60, noise_std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(candidate_angles), size=n_obs)
    true_angles = candidate_angles[idx]
    clean = radon_rows(image, true_angles)
    Y = clean + noise_std * rng.standard_normal(clean.shape)
    return Y, true_angles

def simulate_mixture_data(image1, image2, candidate_angles,
                          n_obs=500, noise_std=0.01, pi=(0.5, 0.5), seed=0):
    rng = np.random.default_rng(seed)
    z = rng.choice(2, size=n_obs, p=pi)
    angle_idx = rng.integers(0, len(candidate_angles), size=n_obs)
    true_angles = candidate_angles[angle_idx]

    Y = []
    for i in range(n_obs):
        image = image1 if z[i] == 0 else image2
        y = radon_rows(image, [true_angles[i]])[0]
        y = y + noise_std * rng.standard_normal(y.shape)
        Y.append(y)

    Y = np.asarray(Y)
    true_classes = z + 1
    return Y, true_classes, true_angles

# Random initialization function
def random_init(size, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((size, size))
    x = (x + np.flipud(x)) / 2.0
    x -= x.min()
    if x.max() > 0:
        x /= x.max()
    return x

# Helper functions for the EM algorithim and the Gibbs sampler
def backproject_single(proj, angle, output_size):
    sino = proj[:, None]
    return iradon(
        sino,
        theta=[angle],
        filter_name=None,
        circle=True,
        output_size=output_size,
    )