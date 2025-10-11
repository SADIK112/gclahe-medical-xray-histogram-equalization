import numpy as np
import cv2
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import validate_image, ensure_grayscale

def global_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Applies Global Histogram Equalization (GHE) to a grayscale image.

    What it does: enhances the contrast of an image by redistributing its pixel
    """