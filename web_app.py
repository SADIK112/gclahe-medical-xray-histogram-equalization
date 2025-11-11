import streamlit as st
import numpy as np
from PIL import Image
import io
import pandas as pd

st.set_page_config(layout="wide", page_title="G-CLAHE Medical Image Enhancement")

from src.utils import ensure_grayscale
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

if 'enhancement_results' not in st.session_state:
    st.session_state.enhancement_results = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

st.sidebar.title('G-CLAHE Controls')
st.sidebar.info('Adjust the parameters for the G-CLAHE algorithm.')

initial_clip_limit = st.sidebar.slider('Initial Clip Limit', 1.0, 10.0, 1.0, 0.5)
tile_size_val = st.sidebar.selectbox('Tile Size', options=[(4, 4), (8, 8), (16, 16), (32, 32)], index=1)
increment = st.sidebar.slider('Clip Limit Increment', 0.5, 5.0, 1.0, 0.5)
similarity_metric = st.sidebar.selectbox('Similarity Metric', ['ssim', 'psnr', 'mse'], index=0)

st.sidebar.title('Comparison Methods')
include_ghe = st.sidebar.checkbox('Include GHE', value=True)
include_clahe = st.sidebar.checkbox('Include CLAHE', value=True)

st.title('G-CLAHE Medical Image Enhancement')
st.write('Upload a medical image (e.g., X-ray) to enhance its contrast and reveal hidden details.')

upload_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if upload_file is not None:
    if st.session_state.uploaded_file != upload_file:
        st.session_state.uploaded_file = upload_file
        image = Image.open(upload_file)
        st.session_state.original_image = np.array(image)
        st.session_state.enhancement_results = None

    original_image = st.session_state.original_image
    grayscale_image = ensure_grayscale(original_image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Original Image')
        st.image(grayscale_image, caption='Original Grayscale Image', use_container_width=True)

    with col2:
        subcol1, subcol2 = st.columns([3, 1])
        with subcol1:
            st.subheader('Enhanced Image')
        with subcol2:
            st.write("")
            run_button = st.button('Run Enhancement')
        if run_button:
            with st.spinner('Applying enhancement algorithms... Please wait.'):
                results = {}
                results['Original'] = grayscale_image

                # G-CLAHE
                gclahe_image, gclahe_metadata = apply_gclahe(
                    grayscale_image,
                    initial_clip_limit=initial_clip_limit,
                    increment=increment,
                    tile_size=tile_size_val,
                    similarity_metric=similarity_metric
                )
                results['G-CLAHE'] = gclahe_image

                # Comparison Methods
                if include_ghe:
                    results['GHE'] = global_histogram_equalization(grayscale_image)
                if include_clahe:
                    results['CLAHE'] = apply_clahe(grayscale_image)

                st.session_state.enhancement_results = {
                    'images': results,
                    'gclahe_metadata': gclahe_metadata
                }
        
        if st.session_state.enhancement_results:
            gclahe_result_image = st.session_state.enhancement_results['images']['G-CLAHE']
            st.image(gclahe_result_image, caption='Enhanced with G-CLAHE', use_container_width=True)

            # Create a downloadable link for the image
            buf = io.BytesIO()
            Image.fromarray(gclahe_result_image).save(buf, format='PNG')
            st.download_button(
                label="Download Enhanced Image",
                data=buf.getvalue(),
                file_name="enhanced_image.png",
                mime="image/png"
            )

# --- Results Tabs ---
if st.session_state.enhancement_results:
    st.markdown('---')
    st.header('Analysis and Comparison')
    tab1, tab2, tab3, tab4 = st.tabs([
        "Slide-by-Slide Comparison",
        "Metrics Analysis",
        'CDFs Pixel Intensity',
        "G-CLAHE Convergence"
    ])

    results = st.session_state.enhancement_results
    images_to_show = results['images']

    with tab1:
        st.subheader('Visual Comparison')
        fig = plot_comparison(images_to_show, save_path=None, show_plot=False)
        st.pyplot(fig)

    with tab2:
        st.subheader('Image Quality Metrics')
        with st.spinner('Calculating metrics...'):
            metrics_data = {name: evaluate_image_quality(img) for name, img in images_to_show.items()}
            metrics_df = pd.DataFrame(metrics_data).T
            st.dataframe(metrics_df)

        st.subheader('Histograms')
        fig = plot_histograms(images_to_show, save_path=None, show_plot=False)
        st.pyplot(fig)

    with tab3:
        st.subheader('Cumulative Distribution Functions (CDFs)')
        fig_cdf = plot_cdf(images_to_show, save_path=None, show_plot=False)
        st.pyplot(fig_cdf)

    with tab4:
        st.subheader('G-CLAHE Iterative Process')
        fig_conv = plot_convergence(results['gclahe_metadata'], save_path=None, show_plot=False)
        st.pyplot(fig_conv)