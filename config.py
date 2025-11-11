import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# G_CLAHE Parameters
G_CLAHE_INITIAL_CLIP_LIMIT = 5.0
G_CLAHE_TILE_SIZE = (32, 32)
G_CLAHE_INCREMENT = 0.5
G_CLAHE_MAX_ITERATIONS = 50

# Standard CLAHE Parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# File path
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
RESULTS_IMAGES_DIR = os.path.join(RESULTS_DIR, 'images')
RESULTS_PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
RESULTS_METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Histogram parameters
HISTOGRAM_SIZE = 256
HISTOGRAM_RANGE = [0, 256]

# Canny edge detection thresholds
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

VIS_DPI = 300
VIS_FIGURE_SIZE = (12, 6)
VIS_COLOR_MAP = 'gray'
VIS_HIST_BINS = 256
VIS_HIST_COLOR = 'blue' 