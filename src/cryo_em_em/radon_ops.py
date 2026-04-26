import numpy as np
from skimage.transform import radon, iradon


def circle_mask(d):
    y, x = np.ogrid[:d, :d]
    cy = cx = d / 2.0
    return (x - cx) ** 2 + (y - cy) ** 2 <= (d // 2) ** 2


def radon_rows(image, angles):
    return radon(image, theta=angles, circle=True).T


def backproject_single(proj, angle, output_size):
    sino = proj[:, None]
    return iradon(
        sino,
        theta=[angle],
        filter_name=None,
        circle=True,
        output_size=output_size,
    )