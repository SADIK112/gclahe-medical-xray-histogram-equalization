import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .utils import ensure_grayscale
from config import HISTOGRAM_SIZE, HISTOGRAM_RANGE, VIS_COLOR_MAP, VIS_FIGURE_SIZE, VIS_HIST_BINS, VIS_DPI

def plot_histograms(images: dict, title: str = 'Histogram Comparison', save_path: str = None, show_plot: bool = True) -> 'plt.Figure':
    """
    Plots images and their corresponding histograms.

    Args:
        images (dict): A dictionary of images to plot.
        title (str): The main title for the figure.
        save_path (str): Optional path to save the figure.
    """
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(n * 6, 8))
    if n == 1:
        axes = np.array([axes]).T # Make it a 2D array for consistent indexing
    fig.suptitle(title, fontsize=16)

    for i, (img_title, img) in enumerate(images.items()):
        # Plot image
        axes[0, i].imshow(img, cmap=VIS_COLOR_MAP)
        axes[0, i].set_title(img_title)
        axes[0, i].axis('off')

        # Plot histogram
        axes[1, i].hist(img.ravel(), bins=VIS_HIST_BINS, range=[0, 256], color='blue', alpha=0.7)
        axes[1, i].set_title(f'{img_title} Histogram')
        axes[1, i].set_xlabel('Pixel Intensity')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=VIS_DPI, bbox_inches='tight')
        print(f"Histogram plot saved to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        return fig
        
    plt.close(fig)

# def plot_histogram(original_image: np.ndarray, enhanced_image: np.ndarray, save_path: str, filename: str):
#     """
#     Plots the histogram of a grayscale image.
#     Args:
#         original_image (np.ndarray): The input grayscale original image.
#         enhanced_image (np.ndarray): The input grayscale enhanced image.
#     """

#     original_image = ensure_grayscale(original_image)
#     enhanced_image = ensure_grayscale(enhanced_image)

#     # Calculate histograms and cumulative frequencies
#     def compute_histogram_data(image):
#         hist = cv2.calcHist([image], [0], None, [HISTOGRAM_SIZE], HISTOGRAM_RANGE)
#         cumsum = hist.cumsum()
#         cumsum_norm = cumsum * float(hist.max()) / cumsum.max()
#         return hist, cumsum_norm

#     orig_hist, orig_cumsum = compute_histogram_data(original_image)
#     enh_hist, enh_cumsum = compute_histogram_data(enhanced_image)

#     # Plot
#     plt.figure(figsize=(12, 5))

#     # Original image histogram
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image Histogram")
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.plot(orig_hist, color='b', label='Histogram')
#     plt.plot(orig_cumsum, color='r', label='Cumulative Frequency')
#     plt.legend()
#     plt.grid()

#     # Enhanced image histogram
#     plt.subplot(1, 2, 2)
#     plt.title("Enhanced Image Histogram")
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.plot(enh_hist, color='b', label='Histogram')
#     plt.plot(enh_cumsum, color='r', label='Cumulative Frequency')
#     plt.legend()
#     plt.grid()

#     plt.tight_layout()

#     # Ensure folder exists
#     save_path = os.path.join(save_path, filename)

#     # Save plot
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Histogram plot saved to: {save_path}")

#     plt.show()

def plot_comparison(images: dict, title: str = "Image comparison", save_path: str = None, show_plot: bool = True):
    """
    Plots a side by side comparison of multiple images

    Args:
        images (dict): A dictionary where keys are titles and values are image arrays.
        title (str): The main title for the entire figure.
        save_path (str): Optional path to save the figure. If None, it will be displayed.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize = 16)

    for ax, (img_title, img) in zip(axes, images.items()):
        ax.imshow(img, cmap = VIS_COLOR_MAP)
        ax.set_title(img_title)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        plt.savefig(save_path, dpi = VIS_DPI, bbox_inches = 'tight')
        print(f"Comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        return fig
    
    plt.close(fig)

def plot_cdf(images: dict, title: str = 'CDF Comparison', save_path: str = None, show_plot: bool = True):
    """
    Plots the Cumulative Distribution Function (CDF) for multiple images on a single graph.

    Args:
        images (dict): A dictionary of images to plot.
        title (str): The main title for the figure.
        save_path (str): Optional path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)

    for img_title, img in images.items():
        hist, _ = np.histogram(img.flatten(), bins=HISTOGRAM_SIZE, range=HISTOGRAM_RANGE)
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        plt.plot(cdf_normalized, label=f'{img_title} CDF')

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 256])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=VIS_DPI, bbox_inches='tight')
        print(f"CDF plot saved to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        return plt.gcf()
        
    plt.close()

def plot_convergence(gclahe_metadata: dict, title: str = 'G-CLAHE Convergence', save_path: str = None, show_plot: bool = True):
    """
    Plots the convergence behavior of the G-CLAHE algorithm.

    Args:
        gclahe_metadata (dict): The metadata dictionary returned by the GCLAHE class.
        title (str): The main title for the figure.
        save_path (str): Optional path to save the figure.
    """
    fig, ax1 = plt.subplots(figsize = VIS_FIGURE_SIZE)
    ax1.set_title(title, fontsize = 16)

    sim_history = gclahe_metadata["similarity_history"]
    clip_history = gclahe_metadata["clip_limit_history"]
    metric_name = gclahe_metadata['similarity_metric'].upper()
    iterations = range(1, len(sim_history) + 1)

    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(f'Similarity ({metric_name})', color = color)
    ax1.plot(iterations, sim_history, marker = 'o', color = color, label=f'Similarity ({metric_name})')
    ax1.tick_params(axis = 'y', labelcolor = color)
    ax1.grid(True, which = 'both', linestyle = "--")

    # Find and mark the best iteration
    higher_is_better = metric_name in ['SSIM', 'PSNR']
    best_idx = np.argmax(sim_history) if higher_is_better else np.argmin(sim_history)
    best_iter = best_idx + 1
    best_sim = sim_history[best_idx]
    ax1.axvline(x=best_iter, color='r', linestyle='--', label=f'Best Iteration ({best_iter})')
    ax1.plot(best_iter, best_sim, 'r*', markersize=15, label=f'Best Score: {best_sim:.4f}')

    # Plot Clip Limit on a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Clip Limit', color=color)
    ax2.plot(iterations, clip_history, 's--', color=color, label='Clip Limit')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=VIS_DPI, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        return fig
        
    plt.close(fig)