import cv2
import numpy as np
import os

from .utils import validate_image, ensure_grayscale
from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE
# from .visualize import plot_histogram

def apply_clahe(
        image: np.ndarray, 
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_grid_size: tuple = CLAHE_TILE_SIZE
    ) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) implementation.
    It enhances local contrast by performing histogram equalization over contextual regions (tiles) of the image

    Args:
        image (np.ndarray): The input grayscale image
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (tuple): Size of grid for histogram equalization. Input image will be divided into 
                                equally sized rectangular tiles 

    Returns:
        np.ndarray: The CLAHE-enhanced 8-bit grayscale image.

    Raises:
        ValueError: If the input image is invalid.
    """

    validate_image(image)
    gray_image = ensure_grayscale(image)

    # OpenCV CLAHE requires uint8 input.
    if gray_image.dtype != np.uint8:
        # assuming float image
        if gray_image.max() <= 1.0:
            gray_image = (gray_image * 255).astype(np.uint8)
        else:
            # just convert to uint8 if any other type
            gray_image = gray_image.astype(np.uint8)

    # handle cases where image is smaller than tile size
    img_h, img_w = gray_image.shape
    tile_h, tile_w = tile_grid_size
    if img_h < tile_h or img_w < tile_w:
        print(f"Warning: Image size ({img_h}x{img_w}) is smaller than tile size ({tile_h}x{tile_w}). CLAHE not applied.")
        return gray_image
    
    try:
        clahe_obj = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_grid_size)
        enhanced_image = clahe_obj.apply(gray_image)
        
        return enhanced_image
    except cv2.error as e:
        print(f"An OpenCV error occurred during CLAHE application: {e}")
        return gray_image
    
class CLAHE:
    """
    A class-based implementation of CLAHE for more structured use.
    Allows setting parameters and then applying the enhancement.
    """
    def __init__(self, clip_limit: float = CLAHE_CLIP_LIMIT, tile_grid_size: tuple = CLAHE_TILE_SIZE):
        """
        Initializes the CLAHE object with specified parameters.

        Args:
            clip_limit (float): Threshold for contrast limiting
            tile_grid_size (tuple): Size of grid for histogram equalization
        """
        if clip_limit <= 0:
            raise ValueError("Clip limit must be a positive number.")
        if not (isinstance(tile_grid_size, tuple) and len(tile_grid_size) == 2):
            raise ValueError("Tile grid size must be a tuple of two positive integers.")

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe_obj = cv2.createCLAHE(clipLimit = self.clip_limit, tileGridSize = self.tile_grid_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the configured CLAHE algorithm to the given image

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The enhanced image
        """
        return apply_clahe(image, self.clip_limit, self.tile_grid_size)
    
# sample test
if __name__ == "__main__":
    from config import INPUT_DIR, OUTPUT_DIR, PLOT_DIR
    from utils import load_image, save_image

    files = os.listdir(PLOT_DIR)
    image_files = [f for f in files if f.lower().endswith(('.jpg'))]

    input_image_path = f"{INPUT_DIR}/x-ray-1.jpg"
    # output_image_path = "x-ray-1-output.jpg"
    output_image_path = f"x-ray-clahe-{len(image_files) + 1}.jpg"

    clahe = CLAHE()
    image = load_image(input_image_path)
    enhanced_image = clahe.apply(image)

    save_image(enhanced_image, output_image_path, OUTPUT_DIR)

    save_file_name = f"x-ray-clahe-hist_{len(image_files) + 1}"
    # plot_histogram(image, enhanced_image, PLOT_DIR, save_file_name)
