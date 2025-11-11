import os
import cv2
import numpy as np
from config import OUTPUT_DIR

def load_image(file_path: str, grayscale: bool = True) -> np.ndarray:
    """
    Loads an image from the specified file path
    
    Args:
        file_path (str): The path of the image file.
        grayscale (bool): If True, loads the image as grayscale
                          If false, loads it in color.

    Returns:
        np.ndarray: The loaded image as a numpy array

    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the image cannot be loaded
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")
    
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(file_path, mode)

    if image is None:
        raise IOError(f"Failed to load image from {file_path}. It may be corrupted or not supported")
    
    return image

def save_image(image: np.ndarray, file_name: str, output_dir: str = OUTPUT_DIR) -> bool:
    """
    Saves a numpy array as an image file

    Args:
        image (np.ndarray): The image to save
        file_name (str): The name of the output file
        output_dir (str): The directory to save the image in

    Returns:
        bool: True if saving is successful, False otherwise
    """
    try:
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, image)
        print(f"Image saved successfully to {output_path}")
        
        return True
    except Exception as e:
        print(f"Error saving image to {os.path.join(output_dir, file_name)}: {e}")
        return False
    
def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Ensure that an image is grayscale

    Args:
        image (np.ndarray): The input image

    Return:
        np.ndarray: The grayscale image
    """
    validate_image(image)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def validate_image(image: np.ndarray):
    """
    Validate the image whether it is a valid numpy array

    Args:
        image (np.ndarray): The image to validate

    Raises:
        ValueError: If the image is not valid numpy array or none or has zero dimensions.
    """

    if image is None or image.size == 0:
        raise ValueError(f"Input image cannot be empty")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be valid numpy array")