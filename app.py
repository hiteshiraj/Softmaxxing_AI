import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Import our backend utilities
from utils.face_detection import FaceDetector
from utils.symmetry import analyze_symmetry
from utils.posture import analyze_posture
from utils.lighting import LightingAnalyzer
from utils.sharpness import analyze_sharpness
from utils.scoring import StyleClusterer, calculate_overall_score
from utils.recommendations import generate_recommendations
from utils.visualization import draw_visualizations
from utils.skin import analyze_skin
from utils.hair import analyze_hair
from utils.advanced_geometry import analyze_facial_geometry, generate_geometry_feedback
from utils.advanced_visualization import draw_advanced_geometry


# Set page config for a premium feel
st.set_page_config(
    page_title="Softmaxxing AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern dark mode aesthetic
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Upload Area styling */
    .stFileUploader > div > div > div > button {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1A1C23;
        border: 1px solid #2D3748;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    
    /* Progress Bars container */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6e8efb, #a777e3);
    }
    
    /* Clean up top padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Cache the initialization of models/detectors to keep the app fast"""
    return {
        'face_detector': FaceDetector(),
        'lighting_analyzer': LightingAnalyzer(),
        'style_clusterer': StyleClusterer()
    }

def create_gauge_chart(score, title):
    """Creates a premium-looking Plotly gauge chart for scores"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'color': 'white', 'size': 18}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#a777e3"},
            'bgcolor': "#1A1C23",
            'borderwidth': 2,
            'bordercolor': "#2D3748",
            'steps': [
                {'range': [0, 50], 'color': '#2d3748'},
                {'range': [50, 80], 'color': '#4a5568'},
                {'range': [80, 100], 'color': '#718096'}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def main():
    st.title("Softmaxxing AI")
    st.markdown("### AI-Powered Presentation & Profile Photo Analysis")
    st.markdown("Upload a selfie to receive constructive, non-toxic feedback on lighting, posture, and image quality.")
    
    models = load_models()
    
    st.markdown("---")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload your photo (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        # Convert PIL to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with st.spinner("Analyzing presentation metrics..."):
            # 1. Face Detection
            rgb_img, landmarks = models['face_detector'].detect_landmarks(image_cv)
            
            if landmarks is None:
                st.error("No face detected. Please upload a clear photo of a single face.")
                return
                
            features = models['face_detector'].extract_features(image_cv, landmarks)
            
            # 2. Heuristics & Analysis
            sym_score, sym_details = analyze_symmetry(features)
            pos_score, pos_details = analyze_posture(features)
            light_score, light_details = models['lighting_analyzer'].analyze(image_cv)
            sharp_score, sharp_details = analyze_sharpness(image_cv)
            skin_score, skin_details = analyze_skin(image_cv, features)
            hair_score, hair_details = analyze_hair(image_cv, features)
            
            scores = {
                'symmetry': sym_score,
                'posture': pos_score,
                'lighting': light_score,
                'sharpness': sharp_score,
                'skin': skin_score,
                'hair': hair_score
            }
            details = {
                'symmetry': sym_details,
                'posture': pos_details,
                'lighting': light_details,
                'sharpness': sharp_details,
                'skin': skin_details,
                'hair': hair_details
            }
            
            # 3. ML Scoring & Recommendations
            style_label = models['style_clusterer'].cluster(image_cv, details)
            overall_score = calculate_overall_score(scores)
            recs = generate_recommendations(scores, details)
            
            # 4. Visualizations
            vis_img = draw_visualizations(image_cv, features, details)
            # Convert BGR back to RGB for displaying in Streamlit
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
            # 5. Advanced Facial Geometry
            geom_metrics = analyze_facial_geometry(features)
            geom_feedback = generate_geometry_feedback(geom_metrics)
            if geom_metrics:
                adv_vis_img = draw_advanced_geometry(image_cv, features)
                adv_vis_img_rgb = cv2.cvtColor(adv_vis_img, cv2.COLOR_BGR2RGB)
            else:
                adv_vis_img_rgb = None

            
        # ==========================================
        # UI Presentation Layer
        # ==========================================
        
        st.markdown("---")
        
        # Top level summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Presentation", f"{overall_score:.1f}/100")
        with col2:
            st.metric("Lighting Quality", light_details['quality_label'])
        with col3:
            st.metric("Detected Style", style_label)
            
        st.markdown("---")
            
        # Side-by-side Image Panels
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("#### Original Photo")
            st.image(image, use_container_width=True)
        with img_col2:
            st.markdown("#### Analysis Overlay")
            st.image(vis_img_rgb, use_container_width=True)
            
        st.markdown("---")
        
        # Detailed Metrics
        st.markdown("### Detailed Metrics")
        
        # Row 1
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.plotly_chart(create_gauge_chart(scores['symmetry'], "Symmetry"), use_container_width=True)
        with metric_col2:
            st.plotly_chart(create_gauge_chart(scores['posture'], "Posture"), use_container_width=True)
        with metric_col3:
            st.plotly_chart(create_gauge_chart(scores['lighting'], "Lighting"), use_container_width=True)
            
        # Row 2
        metric_col4, metric_col5, metric_col6 = st.columns(3)
        with metric_col4:
            st.plotly_chart(create_gauge_chart(scores['sharpness'], "Sharpness"), use_container_width=True)
        with metric_col5:
            st.plotly_chart(create_gauge_chart(scores['skin'], "Skin Quality"), use_container_width=True)
        with metric_col6:
            st.plotly_chart(create_gauge_chart(scores['hair'], "Hair Quality"), use_container_width=True)
            
        st.markdown("---")
        
        # Recommendations
        st.markdown("### Constructive Feedback")
        for rec in recs:
            st.info(rec)
            
        # Advanced Facial Structure Analysis
        if geom_metrics and adv_vis_img_rgb is not None:
            st.markdown("---")
            with st.expander("🧬 Advanced Facial Structure Insights (Optional)"):
                st.info("**Disclaimer**: These measurements are heuristic approximations based on 2D image coordinates. Camera angle, focal length, and lighting heavily distort these metrics. This is an exploratory geometric analysis, NOT a medical or objective assessment of attractiveness.")
                
                adv_col1, adv_col2 = st.columns([1, 1])
                
                with adv_col1:
                    st.markdown("#### Geometric Breakdown")
                    for fb in geom_feedback:
                        st.markdown(f"- {fb}")
                        
                    st.markdown("#### Raw Metrics")
                    st.write(f"**fWHR (Width-to-Height Ratio)**: {geom_metrics['fWHR']:.2f}")
                    st.write(f"**Cheekbone Prominence Ratio**: {geom_metrics['cheekbone_prominence']:.2f}")
                    
                with adv_col2:
                    st.markdown("#### Cyber-Aesthetic Geometry")
                    st.image(adv_vis_img_rgb, use_container_width=True)

if __name__ == "__main__":
    main()
