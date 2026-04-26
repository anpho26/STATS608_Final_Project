import numpy as np
from scipy.ndimage import rotate
from .radon_ops import circle_mask


def random_init(size, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((size, size))
    x = (x + np.flipud(x)) / 2.0
    x -= x.min()
    if x.max() > 0:
        x /= x.max()
    return x


def generate_i(size=32):
    img = np.zeros((size, size), dtype=float)

    c = size // 2
    w = max(2, size // 10)

    img[:, c - w // 2 : c + w // 2 + 1] = 1.0
    img[4:8, size // 4 : 3 * size // 4] = 1.0
    img[-8:-4, size // 4 : 3 * size // 4] = 1.0

    return img


def generate_Sshape(size=32):
    img = np.zeros((size, size), dtype=float)

    w = max(2, size // 10)

    img[4:8, 7:25] = 1.0
    img[14:18, 7:25] = 1.0
    img[24:28, 7:25] = 1.0
    img[4:16, 7:7 + w] = 1.0
    img[16:28, 25 - w:25] = 1.0

    return img


def make_diagonal_stick(size=32, width=2, start=6, end=None):
    if end is None:
        end = size - 6

    img = np.zeros((size, size), dtype=float)

    for i in range(start, end):
        c0 = max(0, i - width // 2)
        c1 = min(size, i + width // 2 + 1)
        img[i, c0:c1] = 1.0

    return img


def make_single_stick(size=32):
    img = np.zeros((size, size), dtype=float)
    for i in range(6, size - 6):
        img[i, i] = 1.0
    return img


def make_double_stick(size=32, separation=3):
    img = np.zeros((size, size), dtype=float)
    for i in range(6, size - 6):
        j1 = i - separation // 2
        j2 = i + separation // 2
        if 0 <= j1 < size:
            img[i, j1] = 1.0
        if 0 <= j2 < size:
            img[i, j2] = 1.0
    return img


def make_two_nonparallel_sticks(size=32, offset=2):
    img = np.zeros((size, size), dtype=float)

    for i in range(6, size - 6):
        j = i - offset
        if 0 <= j < size:
            img[i, j] = 1.0

    c = size - 1 + offset
    for i in range(6, size - 6):
        j = c - i
        if 0 <= j < size:
            img[i, j] = 1.0

    return img


def make_diagonal_rectangle_in_circle(size=32, rect_height=4, rect_width=16, angle=45):
    img = np.zeros((size, size), dtype=float)

    r0 = size // 2 - rect_height // 2
    r1 = r0 + rect_height
    c0 = size // 2 - rect_width // 2
    c1 = c0 + rect_width
    img[r0:r1, c0:c1] = 1.0

    img = rotate(img, angle=angle, reshape=False, order=1)
    img = np.clip(img, 0.0, 1.0)

    mask = circle_mask(size)
    img = np.where(mask, img, 0.0)

    return img