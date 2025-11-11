import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import shannon_entropy as entropy
from .utils import validate_image, ensure_grayscale
from config import CANNY_HIGH_THRESHOLD, CANNY_LOW_THRESHOLD, HISTOGRAM_RANGE, HISTOGRAM_SIZE

# Similarity Metrics

def compute_ssim(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """Computes the Structural Similarity Index (SSIM) between two images."""
    validate_image(imageA)
    validate_image(imageB)
    return ssim(imageA, imageB, data_range=imageA.max() - imageA.min())

def compute_psnr(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    validate_image(imageA)
    validate_image(imageB)
    return psnr(imageA, imageB, data_range=255)

def compute_mse(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """Computes the Mean Squared Error (MSE) between two images."""
    validate_image(imageA)
    validate_image(imageB)
    return mse(imageA, imageB)

def compute_rmse(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """Computes the Root Mean Squared Error (RMSE) between two images."""
    return np.sqrt(compute_mse(imageA, imageB))

def compute_mae(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """Computes the Mean Absolute Error (MAE) between two images."""
    validate_image(imageA)
    validate_image(imageB)
    return np.mean(np.abs(imageA.astype(np.float32) - imageB.astype(np.float32)))

def compute_similarity(imageA: np.ndarray, imageB: np.ndarray, metric: str = 'ssim') -> float:
    """
    A wrapper to compute similarity using a specified metric.

    Args:
        imageA (np.ndarray): The first image.
        imageB (np.ndarray): The second image.
        metric (str): The metric to use ('ssim', 'psnr', 'mse', 'rmse', 'mae').

    Returns:
        float: The similarity score. Note that for 'mse', 'rmse', and 'mae', a lower
               value is better, but this function does not invert it.
    """
    metric = metric.lower()
    if metric == 'ssim':
        return compute_ssim(imageA, imageB)
    elif metric == 'psnr':
        return compute_psnr(imageA, imageB)
    elif metric == 'mse':
        return compute_mse(imageA, imageB)
    elif metric == 'rmse':
        return compute_rmse(imageA, imageB)
    elif metric == 'mae':
        return compute_mae(imageA, imageB)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}. Choose from 'ssim', 'psnr', 'mse', 'rmse', 'mae'.")

# Quality Metrics

def compute_edge_count(image: np.ndarray, low_threshold: int = CANNY_LOW_THRESHOLD, high_threshold: int = CANNY_HIGH_THRESHOLD) -> int:
    """Counts the number of edge pixels using the Canny edge detector."""
    gray_image = ensure_grayscale(image)
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return np.count_nonzero(edges)

def compute_entropy(image: np.ndarray) -> float:
    """Calculates the Shannon entropy of an image."""
    gray_image = ensure_grayscale(image)
    hist, _ = np.histogram(gray_image, bins = HISTOGRAM_SIZE, range = HISTOGRAM_RANGE)
    prob_dist = hist / hist.sum()
    return entropy(prob_dist, base=2)

def compute_average_gradient(image: np.ndarray) -> float:
    """Computes the average magnitude of the image gradient."""
    gray_image = ensure_grayscale(image).astype(np.float32)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    return np.mean(magnitude)

def evaluate_image_quality(image: np.ndarray) -> dict:
    """
    Computes a comprehensive set of quality metrics for a single image.

    Args:
        image (np.ndarray): The image to evaluate.

    Returns:
        dict: A dictionary containing various quality metrics.
    """
    gray_image = ensure_grayscale(image)
    
    metrics = {
        'mean_intensity': np.mean(gray_image),
        'entropy': compute_entropy(gray_image),
        'edge_count': compute_edge_count(gray_image),
        'average_gradient': compute_average_gradient(gray_image)
    }
    return metrics
