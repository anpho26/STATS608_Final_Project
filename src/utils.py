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

    # If RGB or RGBA, keep only RGB channels
    if img.ndim == 3:
        img = rgb2gray(img[:, :, :3])

    img = img.astype(float)

    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    img = resize(
        img,
        (size, size),
        anti_aliasing=True,
        preserve_range=True,
    )

    return img

# Circle masking
def circle_mask(d):
    y, x = np.ogrid[:d, :d]
    cy = cx = (d - 1) / 2.0
    r = d / 2.0
    return (x - cx) ** 2 + (y - cy) ** 2 <= (r-1/2) ** 2

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
    if thickness is None:
        thickness = max(1, size // 12)
    if length is None:
        length = max(4, (size - 3) // 2)
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
        thickness = max(1, size // 12)
    if length is None:
        length = max(4, (size - 3) // 2)

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

def _segment_distance(X, Y, x0, y0, x1, y1):
    """
    Distance from points (X,Y) to the line segment
    joining (x0,y0) and (x1,y1).
    """
    dx = x1 - x0
    dy = y1 - y0
    L2 = dx*dx + dy*dy

    t = ((X - x0)*dx + (Y - y0)*dy) / L2
    t = np.clip(t, 0.0, 1.0)

    px = x0 + t*dx
    py = y0 + t*dy

    return np.sqrt((X - px)**2 + (Y - py)**2)

def generate_7shape(size=64, theta=0, thickness=None):
    d = size
    if thickness is None:
        thickness = max(2, round(d/8))

    # coordinate grid centered at origin
    y, x = np.mgrid[:d, :d]
    x = x - (d-1)/2
    y = y - (d-1)/2

    # rotate coordinates
    t = np.deg2rad(theta)
    xr = np.cos(t)*x + np.sin(t)*y
    yr = -np.sin(t)*x + np.cos(t)*y

    # slightly smaller than the inscribed circle
    R = 0.70 * (d-1)/2

    # vertices
    p0 = (-0.95*R, -0.75*R)   # left end
    p1 = ( 0.95*R, -0.75*R)   # top-right
    p2 = (-0.05*R,  0.10*R)   # elbow
    p3 = (-0.05*R,  1.00*R)   # bottom

    d1 = _segment_distance(xr, yr, *p0, *p1)
    d2 = _segment_distance(xr, yr, *p1, *p2)
    d3 = _segment_distance(xr, yr, *p2, *p3)

    img = (
        (d1 <= thickness/2) |
        (d2 <= thickness/2) |
        (d3 <= thickness/2)
    ).astype(float)

    img *= circle_mask(d)
    return img

def generate_4shape(size=64, theta=0, thickness=None):
    """
    Generate a handwritten-style '4' shape.

    Parameters
    ----------
    d : int
        Image size.
    theta : float
        Rotation angle in degrees.
    thickness : float
        Stroke thickness.
    """

    d = size
    if thickness is None:
        thickness = max(2, round(d/8))

    # coordinate grid centered at origin
    y, x = np.mgrid[:d, :d]
    x = x - (d-1)/2
    y = y - (d-1)/2

    # rotate coordinates
    t = np.deg2rad(theta)
    xr = np.cos(t)*x + np.sin(t)*y
    yr = -np.sin(t)*x + np.cos(t)*y

    # figure size relative to inscribed circle
    R = 0.68 * (d-1)/2

    # vertices describing the "4"
    left_top    = (-0.75*R, -0.95*R)
    left_bottom = (-0.75*R, -0.05*R)

    right_top    = ( 0.45*R, -0.95*R)
    right_bottom = ( 0.45*R,  0.95*R)

    # crossbar height
    cross_left  = left_bottom
    cross_right = (0.45*R, -0.05*R)

    # distances to segments
    d1 = _segment_distance(
        xr, yr,
        *left_top, *left_bottom
    )

    d2 = _segment_distance(
        xr, yr,
        *right_top, *right_bottom
    )

    d3 = _segment_distance(
        xr, yr,
        *cross_left, *cross_right
    )

    img = (
        (d1 <= thickness/2) |
        (d2 <= thickness/2) |
        (d3 <= thickness/2)
    ).astype(float)

    # keep inside the inscribed circle
    img *= circle_mask(d)

    return img

# make "?" and "4" symbols
def make_symbol(symbol="?", size=32, angle=0, canvas_size=256, pad_frac=0.35):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from skimage.transform import resize, rotate
    from matplotlib import font_manager

    # Draw on large black canvas
    img = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)

    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, int(0.85 * canvas_size))

    bbox = draw.textbbox((0, 0), symbol, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = (canvas_size - w) / 2 - bbox[0]
    y = (canvas_size - h) / 2 - bbox[1]

    draw.text((x, y), symbol, fill=255, font=font)

    img = np.asarray(img).astype(float) / 255.0

    # Crop to symbol
    rows, cols = np.where(img > 0)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    img = img[r0:r1 + 1, c0:c1 + 1]

    # Pad enough so rotation does not cut it
    h, w = img.shape
    pad = int(pad_frac * max(h, w))
    img = np.pad(img, pad_width=pad, mode="constant", constant_values=0)

    # Rotate
    if angle != 0:
        img = rotate(
            img,
            angle=angle,
            resize=False,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        )

    # Crop again after rotation
    rows, cols = np.where(img > 1e-6)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    img = img[r0:r1 + 1, c0:c1 + 1]

    # Add small final padding
    final_pad = int(0.15 * max(img.shape))
    img = np.pad(img, final_pad, mode="constant", constant_values=0)

    # Resize to final size
    img = resize(
        img,
        (size, size),
        anti_aliasing=True,
        preserve_range=True,
    )

    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    return img


# Helper functions for the EM algorithm
def radon_rows(image, angles):
    return radon(image, theta=angles, circle=True, preserve_range=True).T

# Simulate
def simulate_data(
    image,
    n_obs=60,
    noise_std=0.01,
    seed=0,
    angle_low=0.0,
    angle_high=360.0,
):
    rng = np.random.default_rng(seed)

    true_angles = rng.uniform(angle_low, angle_high, size=n_obs)

    clean = radon_rows(image, true_angles)
    Y = clean + noise_std * rng.standard_normal(clean.shape)

    return Y, true_angles

def simulate_mixture_data(
    image1,
    image2,
    n_obs=500,
    noise_std=0.01,
    pi=(0.5, 0.5),
    seed=0,
    angle_low=0.0,
    angle_high=360.0,
):
    rng = np.random.default_rng(seed)

    z = rng.choice(2, size=n_obs, p=pi)
    true_angles = rng.uniform(angle_low, angle_high, size=n_obs)

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

