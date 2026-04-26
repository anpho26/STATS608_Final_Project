import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.special import logsumexp
from tqdm.auto import tqdm
from IPython.display import clear_output, display

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon, iradon, resize