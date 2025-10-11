import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import ensure_grayscale
from config import HISTOGRAM_SIZE, HISTOGRAM_RANGE

def plot_histogram(original_image: np.ndarray, enhanced_image: np.ndarray, save_path: str, filename: str):
    """
    Plots the histogram of a grayscale image.
    Args:
        original_image (np.ndarray): The input grayscale original image.
        enhanced_image (np.ndarray): The input grayscale enhanced image.
    """

    original_image = ensure_grayscale(original_image)
    enhanced_image = ensure_grayscale(enhanced_image)

    # Calculate histograms and cumulative frequencies
    def compute_histogram_data(image):
        hist = cv2.calcHist([image], [0], None, [HISTOGRAM_SIZE], HISTOGRAM_RANGE)
        cumsum = hist.cumsum()
        cumsum_norm = cumsum * float(hist.max()) / cumsum.max()
        return hist, cumsum_norm

    orig_hist, orig_cumsum = compute_histogram_data(original_image)
    enh_hist, enh_cumsum = compute_histogram_data(enhanced_image)

    # Plot
    plt.figure(figsize=(12, 5))

    # Original image histogram
    plt.subplot(1, 2, 1)
    plt.title("Original Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(orig_hist, color='b', label='Histogram')
    plt.plot(orig_cumsum, color='r', label='Cumulative Frequency')
    plt.legend()
    plt.grid()

    # Enhanced image histogram
    plt.subplot(1, 2, 2)
    plt.title("Enhanced Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(enh_hist, color='b', label='Histogram')
    plt.plot(enh_cumsum, color='r', label='Cumulative Frequency')
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Ensure folder exists
    save_path = os.path.join(save_path, filename)

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Histogram plot saved to: {save_path}")

    plt.show()


