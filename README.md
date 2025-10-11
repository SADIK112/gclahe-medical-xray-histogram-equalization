# G-CLAHE: Guided Contrast Limited Adaptive Histogram Equalization for Medical Image Enhancement

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a Python implementation of the G-CLAHE algorithm, an advanced technique for enhancing the contrast of medical images, particularly X-rays. The algorithm iteratively applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and uses a globally equalized version of the image as a guide to determine the optimal level of local enhancement.

This approach prevents the over-amplification of noise often seen with standard CLAHE while effectively revealing subtle details in low-contrast regions.

## Key Features

- **Core G-CLAHE Algorithm**: An iterative, guided approach to find the optimal CLAHE clip limit.
- **Multiple Similarity Metrics**: Use SSIM, PSNR, or MSE to guide the optimization process.
- **Comparison Methods**: Automatically compare results against standard GHE and CLAHE.
- **Comprehensive Evaluation**: Calculates key quality metrics (Entropy, Edge Count, Average Gradient) for quantitative analysis.
- **Rich Visualizations**: Generates side-by-side image comparisons, histograms, CDF plots, and convergence graphs.
- **Interactive Web App**: A Streamlit-based interface for easy, interactive enhancement and parameter tuning.
- **Command-Line Tools**: Includes scripts for single-image (`main.py`) and batch processing (`batch_process.py`).

## Project Structure

```
gclahe-medical-enhancement/
│
├── src/                # Source code for all algorithms and utilities
│   ├── gclahe.py       # The main G-CLAHE algorithm
│   ├── clahe.py        # Wrapper for standard CLAHE
│   ├── ghe.py          # Global Histogram Equalization
│   ├── metrics.py      # Image quality and similarity metrics
│   ├── visualize.py    # Plotting and visualization functions
│   └── utils.py        # I/O and image validation helpers
│
├── data/
│   ├── input/          # Place your input images here for batch processing
│   └── sample/         # Contains a sample X-ray image
│
├── tests/              # Unit tests for all core modules
│
├── main.py             # CLI for processing a single image
├── batch_process.py    # CLI for processing all images in a directory
├── web_app.py          # Interactive Streamlit web application
├── config.py           # Central configuration file for all parameters
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd gclahe-medical-enhancement
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Quick Start

There are three ways to use this project: the interactive web app, the single-image script, or the batch processor.

### 1. Using the Streamlit Web App (Recommended)

This is the easiest way to get started. Run the following command in your terminal:

```bash
streamlit run web_app.py
```

This will open a new tab in your browser with the application interface. You can upload an image, adjust parameters in the sidebar, and see the results in real-time.

![Streamlit App Screenshot](https://user-images.githubusercontent.com/path/to/screenshot.png) <!-- Add a screenshot later -->

### 2. Processing a Single Image

Use `main.py` to process a single image and generate a full report, including enhanced images, plots, and a metrics CSV file.

```bash
python main.py data/sample/sample_xray.jpg
```

**Optional arguments:**

-   `--metric`: Similarity metric (`ssim`, `psnr`, `mse`). Default is `ssim`.
-   `--clip_limit`: Initial clip limit. Default is `1.0`.
-   `--increment`: Clip limit increment per iteration. Default is `1.0`.

Example:
```bash
python main.py data/sample/sample_xray.jpg --metric psnr --clip_limit 2.0
```

Results will be saved in the `results/images/` and `results/plots/` directories, organized by the input image's name.

### 3. Batch Processing a Directory

Place all your images in the `data/input/` directory and run `batch_process.py`.

```bash
python batch_process.py
```

This will process all images in the specified input directory, save the enhanced versions, and generate a summary CSV report in `results/metrics/`.

## Running Tests

To ensure all components are working correctly, run the unit tests:

```bash
python -m unittest discover tests
```

## How It Works

The G-CLAHE algorithm is based on the idea that the global contrast characteristics of an image are desirable, but local details need enhancement. The algorithm works as follows:

1.  **Create a Global Reference (GEI)**: Apply Global Histogram Equalization (GHE) to the original image. This `GEI` serves as the "target" for global appearance.
2.  **Iterative Enhancement**: Start with a low clip limit and apply CLAHE to the image, creating a Locally Enhanced Image (`LEI`).
3.  **Guided Comparison**: Apply GHE to the `LEI` to get `GHE(LEI)`. Compare this `GHE(LEI)` with the `GEI` using a similarity metric (e.g., SSIM).
4.  **Converge**: If the similarity improves, keep the `LEI` and increase the clip limit for the next iteration. If similarity worsens, the optimal point has been passed, and the loop terminates.
5.  **Return Best Result**: The `LEI` from the iteration with the highest similarity score is returned as the final enhanced image.
