import argparse
import os
import time
import pandas as pd

from src.utils import load_image, save_image
from src.ghe import global_histogram_equalization
from src.clahe import apply_clahe
from src.gclahe import apply_gclahe
from src.metrics import evaluate_image_quality
from src.visualize import (
    plot_histograms,
    plot_comparison,
    plot_cdf,
    plot_convergence
)
from config import (
    RESULTS_IMAGES_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_PLOTS_DIR,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE
)

def main():
    """
    Main function to run the G-CLAHE enhancement process on a single image.
    """
    parser = argparse.ArgumentParser(description='G-CLAHE Medical Image Enhancement')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('--metric', type=str, default='ssim', help='Similarity metric for G-CLAHE (ssim, psnr, mse).')
    parser.add_argument('--clip_limit', type=float, default=1.0, help='Initial clip limit for G-CLAHE.')
    parser.add_argument('--increment', type=float, default=1.0, help='Clip limit increment for G-CLAHE.')
    parser.add_argument('--tile_size', type=int, nargs=2, default=[8, 8], help='Tile grid size for G-CLAHE.')
    args = parser.parse_args()
    print(f"what is this? {args}")
    print("--- G-CLAHE Medical Image Enhancement ---")
    print(f"Processing image: {args.image_path}")
    print(f"Parameters: Metric={args.metric}, Initial Clip Limit={args.clip_limit}, Increment={args.increment}, Tile Size={tuple(args.tile_size)}")

    try:
        original_image = load_image(args.image_path, grayscale = True)
        print(f"Image loaded successfully. Dimensions: {original_image.shape}")

        # --- Apply Enhancement Algorithms ---
        start_time = time.time()

        # 1. G-CLAHE
        print("\nApplying G-CLAHE...")
        gclahe_image, gclahe_metadata = apply_gclahe(
            original_image,
            initial_clip_limit = args.clip_limit,
            increment = args.increment,
            tile_size = tuple(args.tile_size),
            similarity_metric = args.metric
        )
        print(f"G-CLAHE finished in {gclahe_metadata['iterations']} iterations. Best Clip Limit: {gclahe_metadata['best_clip_limit']:.2f}")

        # 2. Standard CLAHE
        print("Applying standard CLAHE...")
        clahe_image = apply_clahe(original_image, clip_limit = CLAHE_CLIP_LIMIT, tile_grid_size = CLAHE_TILE_SIZE)

        # 3. GHE
        print("Applying Global Histogram Equalization (GHE)...")
        ghe_image = global_histogram_equalization(original_image)

        print(f"\nAll enhancement methods applied in {time.time() - start_time:.2f} seconds.")

        # --- Evaluate Metrics ---
        print('\nEvaluating image quality metrics')
        images_to_evaluate = {
            'Original': original_image,
            'G-CLAHE': gclahe_image,
            'CLAHE': clahe_image,
            'GHE': ghe_image
        }

        metrics_results = {name: evaluate_image_quality(img) for name, img in images_to_evaluate.items()}

        # --- Display and Save Metrics
        metrics_df = pd.DataFrame(metrics_results).T
        print("\nImage Quality Metrics:")
        print(metrics_df.to_string())

        # --- Save Metrics to CSV ---
        file_basename = os.path.splitext(os.path.basename(args.image_path))[0]
        os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
        metrics_csv_path = os.path.join(RESULTS_METRICS_DIR, f"{file_basename}_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=True)
        print("\nMetrics saved tp {metrics_csv_path}")

        # --- Saved Enhanced Image ---
        print("\nSaving enhanced image...")
        image_save_dir = os.path.join(RESULTS_IMAGES_DIR, file_basename)
        save_image(gclahe_image, 'gclahe.png', output_dir = image_save_dir)
        save_image(clahe_image, 'clahe.png', output_dir = image_save_dir)
        save_image(ghe_image, 'ghe.png', output_dir = image_save_dir)
        print(f"Enhanced images saved to {image_save_dir}")

        # --- Generate and Save Visualization ---
        print("\nGenerating and saving visualizations...")
        plots_save_dir = os.path.join(RESULTS_PLOTS_DIR, file_basename)

        # 1. Comparison Plot
        plot_comparison(images_to_evaluate, save_path = os.path.join(plots_save_dir, 'comparison.png'))
        # 2. Histogram Plot
        plot_histograms(images_to_evaluate, save_path = os.path.join(plots_save_dir, 'histograms.png'))
        # 3. CDF Plot
        plot_cdf(images_to_evaluate, save_path = os.path.join(plots_save_dir, 'cdfs.png'))
        # 4. Convergence Plot for G-CLAHE
        plot_convergence(gclahe_metadata, save_path=os.path.join(plots_save_dir, 'gclahe_convergence.png'))
        
        print(f"\n--- Processing Complete ---")
        print(f"All results saved in '{RESULTS_IMAGES_DIR}' and '{RESULTS_PLOTS_DIR}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()