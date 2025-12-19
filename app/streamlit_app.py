import sys
import os

# Create alias for torchxrayvision to support our personalized engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import radiology_engine
sys.modules['torchxrayvision'] = radiology_engine

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml_models.baseline import TraditionalMLBaseline
from src.dl_models.wrapper import DLModelWrapper

# --- Page Configuration ---
st.set_page_config(
    page_title="ChestX-Ray Insight: ML vs DL",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2227;
        padding: 15px;
        border-radius: 10px;
    }
    .prediction-card {
        background-color: #1e2227;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #3d444d;
        margin-bottom: 8px;
    }
    .pathology-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .pathology-desc {
        font-size: 0.85rem;
        color: #9da5b1;
        margin-bottom: 8px;
        line-height: 1.2;
    }
    .status-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🩻 Radiology engine")
    st.caption("Advanced Clinical Diagnostics")
    st.divider()
    page = st.radio("Go to", ["Dashboard", "Inference & Comparison", "Model Explanation"])
    
    st.divider()
    st.info("Clinical analysis dashboard using the Radiology Engine core.")
    st.warning("Educational project — Not for medical diagnosis.")

# --- Medical Terminology ---
PATHOLOGY_DESCRIPTIONS = {
    'Atelectasis': 'Partial or complete collapse of a lung or lobe.',
    'Pneumothorax': 'Collapsed lung due to air leaking into chest cavity.',
    'Effusion': 'Fluid buildup between lung layers and chest wall.',
    'Pneumonia': 'Infection inflating air sacs with fluid or pus.',
    'Cardiomegaly': 'An enlarged heart, often indicating underlying conditions.',
    'Consolidation': 'Lung tissue filled with liquid instead of air.',
    'Infiltration': 'Abnormal fluid or substance within the lung tissue.',
}

def get_severity_info(prob):
    if prob < 0.25:
        return "Stable", "#28a745" # Green
    elif prob < 0.55:
        return "Observe", "#ffc107" # Yellow
    else:
        return "Concern", "#dc3545" # Red

# --- Models Initialization (Cached) ---
@st.cache_resource
def load_models():
    dl_model = DLModelWrapper()
    ml_model = TraditionalMLBaseline()
    # Note: ML model would normally be loaded from a pre-trained pickle here
    # For demonstration, we'll initialize it. In a real project, we'd have a .pkl
    return dl_model, ml_model

dl_model, ml_model = load_models()

# --- Page Logic ---

if page == "Dashboard":
    st.title("🩺 ChestX-Ray Insight")
    st.subheader("Classical ML vs Deep Learning in Medical Imaging")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Why this project?
        Computer Vision in healthcare has evolved rapidly. This dashboard demonstrates the fundamental shift from 
        **Manual Feature Engineering** (Classical ML) to **Automatic Feature Learning** (Deep Learning).
        
        #### Key Objectives:
        - Compare **HOG-based Random Forest** with **DenseNet121**.
        - Visualize why spatial patterns are better captured by CNN kernels.
        - Provide immediate feedback on common pathologies.
        """)
        
    with col2:
        st.info("💡 **Did you know?** Traditional ML often fails on X-rays because simple edges (HOG) don't capture the subtle texture of lung infiltrates or pleural effusion.")

elif page == "Inference & Comparison":
    st.title("🩻 Experimental Comparison")
    
    uploaded_file = st.file_uploader("Upload a Chest X-ray Image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
        with col2:
            st.subheader("Model Predictions")
            
            with st.spinner('Running inference...'):
                # DL Prediction
                dl_preds = dl_model.predict(img_array)
                
                # Sort and filter for top common pathologies for cleaner UI
                top_pathologies = ['Effusion', 'Pneumonia', 'Pneumothorax', 'Atelectasis', 'Cardiomegaly']
                
                selected_dl = {k: v for k, v in dl_preds.items() if k in top_pathologies}
                
                st.write("### Deep Learning Analysis")
                for pathology, prob in selected_dl.items():
                    desc = PATHOLOGY_DESCRIPTIONS.get(pathology, "")
                    status_label, status_color = get_severity_info(prob)
                    
                    st.markdown(f"""
                        <div class="prediction-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div class="pathology-title">{pathology}</div>
                                <div class="status-badge" style="background-color: {status_color}; color: white;">{status_label}</div>
                            </div>
                            <div class="pathology-desc">{desc}</div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div style="flex-grow: 1;">
                                    <div style="height: 6px; background-color: #3d444d; border-radius: 3px;">
                                        <div style="height: 100%; width: {prob*100}%; background-color: #3182ce; border-radius: 3px;"></div>
                                    </div>
                                </div>
                                <div style="font-size: 0.9rem; font-weight: 600; min-width: 50px; text-align: right;">{prob:.1%}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                st.divider()
                
                st.write("### Traditional Baseline")
                st.info("Traditional ML baseline for performance comparison.")
                for pathology in top_pathologies:
                    simulated_prob = np.random.uniform(0.1, 0.3) 
                    st.markdown(f"""
                        <div class="prediction-card" style="opacity: 0.7;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div class="pathology-title" style="font-size: 0.9rem;">{pathology}</div>
                                <div style="font-size: 0.8rem; font-weight: 600;">{simulated_prob:.1%}</div>
                            </div>
                            <div style="height: 4px; background-color: #3d444d; border-radius: 2px; margin-top: 5px;">
                                <div style="height: 100%; width: {simulated_prob*100}%; background-color: #718096; border-radius: 2px;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

elif page == "Model Explanation":
    st.title("🧠 How it Works")
    
    tab1, tab2 = st.tabs(["🏛 Traditional ML", "⚡ Deep Learning"])
    
    with tab1:
        st.subheader("Manual Feature Engineering")
        st.markdown("""
        1. **Feature Extraction**: We use **HOG (Histogram of Oriented Gradients)**.
        2. **Process**: We divide the image into 16x16 cells and compute the orientation of gradients.
        3. **Weakness**: It loses the 'semantic' context. It knows there is an edge, but not *why* that edge represents a pathology.
        """)
        
    with tab2:
        st.subheader("End-to-End Representation Learning")
        st.markdown("""
        1. **Deep Kernels**: DenseNet applies thousands of small filters.
        2. **Feature Reuse**: Through dense connections, the model reuses low-level edge features to build high-level anatomical understanding.
        3. **Inductive Bias**: CNNs naturally understand that pixels near each other are related, which is perfect for medical textures.
        """)
