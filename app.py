import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration to wide layout and custom title
st.set_page_config(page_title="AIVid-Optimizer Dashboard v2", layout="wide")

# -------------------------------------------------------------
# PREMIUM INTUITIVE UI: CUSTOM CSS INJECTION
# -------------------------------------------------------------
st.markdown("""
    <style>
    /* Main Background & Text Color */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    /* FIX FOR LABELS - Text color white and clear */
    .stSlider label, .stSelectbox label, p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    /* Custom Container Cards with Glassmorphism effect */
    .metric-card {
        background-color: #1F2937;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }

        /* Fix to Align Right Panel Title with Left Panel Controls perfectly */
    .stMarkdown h3 {
        margin-top: 0px !important;
        padding-top: 0px !important;
        line-height: 1.2 !important;
    }


    .cost-card {
        border-left: 5px solid #10B981;
    }
    /* Animated Hover Effect for Action Button */
    div.stButton > button:first-child {
        background-color: #3B82F6;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# 1. Load the upgraded pre-trained ML engines
with open('time_model.pkl', 'rb') as f:
    time_model = pickle.load(f)
with open('cost_model.pkl', 'rb') as f:
    cost_model = pickle.load(f)
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# 2. Design Dashboard Layout (Header)
st.title("🚀 AIVid-Optimizer v2.0")
st.markdown("💻 **Predictive Latent-Space Infrastructure Gateway with 4-Resolution Support**")
st.markdown("---")

# 3. Create Side-by-Side Layout using Streamlit Columns
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 🎛️ Infrastructure Configuration")
    with st.container():
        video_length = st.slider("Select Video Length (Seconds):", min_value=5, max_value=60, value=15)
        
        # UPGRADED: Added 4 distinct options
        resolution = st.selectbox("Select Target Resolution:", 
                                  options=["720p HD Ready", "1080p Full HD", "2K Quad HD", "4K Ultra HD"])
        
        # UPGRADED: Added 120 and 144 FPS options
        fps = st.selectbox("Select Target Frame Rate (FPS):", options=[24, 30, 60, 120, 144])
        
        # Map resolution text string to numerical encoder tier matching the dataset
        res_mapping = {"720p HD Ready": 0, "1080p Full HD": 1, "2K Quad HD": 2, "4K Ultra HD": 3}
        resolution_tier = res_mapping[resolution]
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.button("Analyze Compute Profile")

with col2:
    
    st.markdown("### 📊 Live Performance & Routing Analytics")
    
    if submit_btn:
        # Structure the input data with the EXACT feature names used in training
        input_data = pd.DataFrame({
            'video_length_sec': [video_length],
            'resolution_tier': [resolution_tier], # Fixed Feature Name
            'fps': [fps]
        })
        
        # Run predictions and force float conversion to avoid format errors
        pred_time = float(time_model.predict(input_data)[0])
        pred_cost = float(cost_model.predict(input_data)[0])
        
        # Infrastructure routing layer via KMeans
        cluster_input = [[pred_time, pred_cost]]
        tier_label = int(kmeans_model.predict(cluster_input)[0])
        
        # Strategic Business Mappings
        tier_mapping = {
            0: ("Eco-Friendly / Consumer GPUs (RTX 4090)", "success"),
            1: ("Standard Enterprise Load (NVIDIA A10G Cluster)", "warning"),
            2: ("Heavy Compute Cluster (NVIDIA H100 / Premium Cloud Required)", "error")
        }
        tier_text, tier_style = tier_mapping[tier_label]
        
        # Display the custom metrics cards
        st.markdown(f"""
            <div class="metric-card">
                <span style='color: #9CA3AF; font-size: 14px; font-weight: bold;'>ESTIMATED GPU RENDER TIME</span>
                <h1 style='color: #3B82F6; margin: 0; font-size: 36px;'>{pred_time:.2f} Seconds</h1>
            </div>
            <div class="metric-card cost-card">
                <span style='color: #9CA3AF; font-size: 14px; font-weight: bold;'>ESTIMATED INFRASTRUCTURE COST</span>
                <h1 style='color: #10B981; margin: 0; font-size: 36px;'>${pred_cost:.2f} USD</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Display localized routing warnings
        if tier_style == "error":
            st.error(f"🚨 **Routing Matrix Notice:** {tier_text}")
        elif tier_style == "warning":
            st.warning(f"⚠️ **Routing Matrix Notice:** {tier_text}")
        else:
            st.success(f"✅ **Routing Matrix Notice:** {tier_text}")
    else:
        st.info("Adjust configurations on the left panel and click 'Analyze Compute Profile' to trigger live inference models.")
