import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize


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