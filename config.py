import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# G_CLAHE Parameters
G_CLAHE_INITIAL_CLIP_LIMIT = 1.0
G_CLAHE_TILE_SIZE = (8, 8)
G_CLAHE_INCREMENT = 1.0
G_CLAHE_MAX_ITERATIONS = 100

# Standard CLAHE Parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# File path
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')

# Histogram parameters
HISTOGRAM_SIZE = 256
HISTOGRAM_RANGE = [0, 256]