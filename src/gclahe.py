import numpy as np
import os

from .utils import validate_image, ensure_grayscale
from config import (
    G_CLAHE_INITIAL_CLIP_LIMIT,
    G_CLAHE_INCREMENT,
    G_CLAHE_MAX_ITERATIONS,
    G_CLAHE_TILE_SIZE,
)
from .ghe import global_histogram_equalization
from .clahe import apply_clahe
from .metrics import compute_similarity
# from .visualize import plot_histogram


class GCLAHE:
    """
    Class to perform Global Contrast Limited Adaptive Histogram Equalization (G-CLAHE)
    with iterative clip limit adjustment.
    """

    def __init__(
        self,
        initial_clip_limit: float = G_CLAHE_INITIAL_CLIP_LIMIT,
        tile_size: tuple = G_CLAHE_TILE_SIZE,
        increment: float = G_CLAHE_INCREMENT,
        max_iterations: int = G_CLAHE_MAX_ITERATIONS,
        similarity_metric: str = "ssim",
    ):
        """
        Initialize the G-CLAHE processor.

        Args:
            initial_clip_limit (float): Starting clip limit for CLAHE.
            tile_size (tuple): Size of the tiles for CLAHE.
            increment (float): Step size to increase the clip limit per iteration.
            max_iterations (int): Maximum iterations for the optimization loop.
            similarity_metric (str): Similarity metric for convergence check.
                                     Options: "ssim", "mse", "psnr".
        """
        self.initial_clip_limit = initial_clip_limit
        self.tile_size = tile_size
        self.increment = increment
        self.max_iterations = max_iterations
        self.similarity_metric = similarity_metric.lower()

        # Determine if higher or lower metric score is better
        self.higher_is_better = self.similarity_metric in ["ssim", "psnr"]

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply the G-CLAHE algorithm to an image.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            tuple[np.ndarray, dict]: Enhanced image and process metadata.
        """
        # Preprocess input image
        validate_image(image)
        original_image = ensure_grayscale(image)

        # Compute Global Enhanced Image (GEI) as the reference
        gei = global_histogram_equalization(original_image)

        # Initialize variables
        current_clip_limit = self.initial_clip_limit
        best_similarity = -np.inf if self.higher_is_better else np.inf
        best_enhanced_image = original_image.copy()
        lei = original_image.copy()  # First LEI

        history = {"similarity": [], "clip_limit": []}

        # Iterative optimization
        for i in range(self.max_iterations):
            # Apply CLAHE
            current_lei = apply_clahe(
                lei, clip_limit=current_clip_limit, tile_grid_size=self.tile_size
            )

            # Apply GHE for comparison
            ghe_lei = global_histogram_equalization(current_lei)

            # Compute similarity
            current_similarity = compute_similarity(
                ghe_lei, gei, metric=self.similarity_metric
            )

            history["similarity"].append(current_similarity)
            history["clip_limit"].append(current_clip_limit)

            # Convergence check
            converged = False
            if self.higher_is_better:
                if current_similarity > best_similarity:
                    best_similarity = current_similarity
                    best_enhanced_image = current_lei
                    lei = current_lei
                else:
                    converged = True
            else:
                if current_similarity < best_similarity:
                    best_similarity = current_similarity
                    best_enhanced_image = current_lei
                    lei = current_lei
                else:
                    converged = True

            if converged:
                print(f"Convergence reached at iteration {i+1}. Similarity stopped improving.")
                break

            current_clip_limit += self.increment
        else:
            print(f"Max iterations ({self.max_iterations}) reached.")

        # Prepare metadata
        metadata = {
            "best_clip_limit": (
                history["clip_limit"][np.argmax(history["similarity"])]
                if self.higher_is_better
                else history["clip_limit"][np.argmin(history["similarity"])]
            ),
            "best_similarity": best_similarity,
            "iterations": len(history["clip_limit"]),
            "similarity_history": history["similarity"],
            "clip_limit_history": history["clip_limit"],
            "similarity_metric": self.similarity_metric,
        }
        return best_enhanced_image, metadata

def apply_gclahe(image: np.ndarray, **kwargs) -> tuple[np.ndarray, dict]:
    """
    Static helper to apply G-CLAHE to an image directly.
    """
    gclahe_processor = GCLAHE(**kwargs)
    return gclahe_processor.apply(image)

if __name__ == "__main__":
    from config import INPUT_DIR, OUTPUT_DIR, PLOT_DIR
    from utils import load_image, save_image
    files = os.listdir(PLOT_DIR)
    image_files = [f for f in files if f.lower().endswith(".png")]

    input_image_path = f"{INPUT_DIR}/x-ray-1.jpg"
    # output_image_path = "x-ray-3-output.jpg"
    output_image_path = f"x-ray-gclahe-{len(image_files) + 1}.jpg"

    gclahe = GCLAHE()
    image = load_image(input_image_path)
    enhanced_image, metadata = gclahe.apply_gclahe(image)

    save_image(enhanced_image, output_image_path, OUTPUT_DIR)

    save_file_name = f"x-ray-gclahe-hist_{len(image_files) + 1}"
    # plot_histogram(image, enhanced_image, PLOT_DIR, save_file_name)
