import numpy as np
import cv2
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import validate_image, ensure_grayscale
from config import HISTOGRAM_SIZE, HISTOGRAM_RANGE
from visualize import plot_histogram

def global_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Applies Global Histogram Equalization (GHE) to a grayscale image.

    What it does: enhances the contrast of an image by redistributing its pixel intensities.
    The core idea is to create a new image where the cumulative distribution (CDF) is linear,
    effectively spreading out the most frequent intensity values.

    Args:
        image (np.ndarray): The input image as a numpy array. Can be color or grayscale.
                            If it is color will be converted to grayscale

    Returns:
        np.ndarray: The contrast-enhanced, 8-bit grayscale image.

    Raises:
        ValueError: If the input image is empty or invalid.
    """

    validate_image(image)
    gray_image = ensure_grayscale(image)

    # Handle edge case: image with uniform intensity (eg., all black or white)
    if np.all(gray_image == gray_image[0, 0]):
        return gray_image.copy()
    
    # Histogram computation
    # Calculating frequency of each intensity level (0-255)
    hist, _ = np.histogram(gray_image.flatten(), bins = HISTOGRAM_SIZE, range = HISTOGRAM_RANGE)
    # CDF calculation (computing the cumulative sum of histogram)
    cdf = hist.cumsum()
    # CDF Normalization
    # Finding the minimum non-zero value in the CDF for normalization
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # Fill the masked values with 0
    cdf_normalized = np.ma.filled(cdf_m, 0).astype('uint8')
    # Apply transformation (using the normalized CDF as a lookup table to map pixel values)
    enhanced_image = cdf_normalized[gray_image]

    return enhanced_image

class GHE:
    """
    Class to perform Global Histogram Equalization (GHE) with iterative clip limit adjustment.
    """
    def __init__(self):
        pass
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the GHE algorithm to the given image

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The enhanced image.
        """
        return global_histogram_equalization(image)

if __name__ == "__main__":
    from config import INPUT_DIR, OUTPUT_DIR, PLOT_DIR
    from utils import load_image, save_image

    input_image_path = f"{INPUT_DIR}/x-ray-1.jpg"
    output_image_path = "x-ray-1-output.jpg"

    ghe = GHE()
    image = load_image(input_image_path)
    enhanced_image = ghe.apply(image)

    save_image(enhanced_image, output_image_path, OUTPUT_DIR)
    files = os.listdir(PLOT_DIR)
    image_files = [f for f in files if f.lower().endswith(('.jpg'))]
    save_file_name = f"x-ray-ghe-hist_{len(image_files) + 1}"
    plot_histogram(image, enhanced_image, PLOT_DIR, save_file_name)