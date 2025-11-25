"""
Smart Search Engine for Farmers - Streamlit Application
Version: 3.0 (Full Features + Fixed UI Colors + Centered Audio)
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import tempfile
from audio_recorder_streamlit import audio_recorder

# -----------------------------------------------------------------------------
# IMPORT ERROR HANDLING
# -----------------------------------------------------------------------------
try:
    from gemini_helper import GeminiHelper
    from audio_processor import AudioProcessor
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except ImportError as e:
    st.error(f"❌ Critical Error: Missing required modules. {str(e)}")
    st.info("Please ensure 'gemini_helper.py', 'audio_processor.py', and 'requirements.txt' dependencies are installed.")
    st.stop()

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Farmer Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. CSS & STYLING (THEME & FIXES)
# -----------------------------------------------------------------------------
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <style>
    /* ---------------- GLOBAL VARIABLES ---------------- */
    :root {
        --primary-green: #4CAF50;
        --dark-green: #2E7D32;
        --light-green: #8BC34A;
        --text-dark: #333333;
        --bg-gradient: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* ---------------- RESET & FONTS ---------------- */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: var(--text-dark);
    }
    
    /* Background */
    .stApp {
        background: var(--bg-gradient);
    }
    
    /* Hide Default Header/Footer/Hamburger */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Padding Adjustments */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }

    /* ---------------- FIX 1: TEXT INPUTS (Make Text Black & Visible) ---------------- */
    /* Force White Background and Black Text for Text Areas and Inputs */
    .stTextArea textarea, .stTextInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #E0E0E0 !important;
        border-radius: 12px !important;
        caret-color: #000000;
    }
  
    
    /* Focus State - Green Border */
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2) !important;
    }
    
    /* Labels styling */
    .stTextArea label p, .stTextInput label p {
        font-size: 1.1rem !important;
        color: #2E7D32 !important;
        font-weight: 600 !important;
    }

    /* ---------------- FIX 2: FILE UPLOADER (Make Text Black) ---------------- */
    /* Target the container and all internal text elements */
    [data-testid="stFileUploader"] {
        padding: 2rem;
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        background-color: #F1F8F4; /* Light green tint background */
        text-align: center;
    }
    
    /* Force specific text elements to black */
    [data-testid="stFileUploader"] div div::before {color: white !important;} 
    [data-testid="stFileUploader"] div div::after {color: white !important;}
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] small, 
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div {
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] svg {
        fill: #ffffff !important;
    }
    
    /* Remove default Streamlit background on the dropzone */
    [data-testid="stFileUploadDropzone"] {
        background-color: transparent !important;
    }

    /* ---------------- FIX 3: AUDIO RECORDER CENTERING ---------------- */
    /* Align iframes (used by custom components) to center */
    iframe {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* ---------------- HEADER & HERO SECTIONS ---------------- */
    .custom-header {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 1rem 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .custom-header h1 {
        color: white !important;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .custom-header i {
        color: white !important;
        font-size: 1.8rem;
    }

    .hero-section {
        text-align: center;
        padding: 3rem 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2E7D32 !important;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #555 !important;
        max-width: 800px;
        margin: 0 auto 3rem auto;
    }
    
    /* ---------------- FEATURE CARDS ---------------- */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.3s;
        border-bottom: 4px solid #4CAF50;
        height: 100%;
    }
    .feature-card:hover { transform: translateY(-5px); }
    .feature-icon { font-size: 2.5rem; color: #4CAF50 !important; margin-bottom: 1rem; }
    .feature-card h3 { color: #2E7D32 !important; font-size: 1.2rem; margin-bottom: 0.5rem; font-weight: 600; }
    .feature-card p { color: #666 !important; font-size: 0.9rem; }

    /* ---------------- CONTENT CONTAINER ---------------- */
    .content-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin: 1rem auto;
        border: 1px solid #eee;
        max-width: 1200px;
    }
    
    .section-header-icon { font-size: 3rem; color: #4CAF50 !important; display: block; margin: 0 auto 1rem auto; }
    .section-title { color: #2E7D32 !important; text-align: center; margin-bottom: 0.5rem; font-weight: 700; font-size: 2rem; }
    .section-subtitle { color: #666 !important; text-align: center; margin-bottom: 2rem; font-size: 1.1rem; }

    /* ---------------- TABS & BUTTONS ---------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        margin-bottom: 1rem;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 30px;
        padding: 0.8rem 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.3);
    }
    .stTabs p { font-size: 1.1rem; font-weight: 600; color: #555; }
    .stTabs [aria-selected="true"] p { color: white !important; }

    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.2) !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(76, 175, 80, 0.3) !important;
    }
    .stButton > button p { color: white !important; font-size: 1.1rem; }

    /* ---------------- RESULT BOXES ---------------- */
    /* Diagnosis (Orange) */
    .diagnosis-box {
        background-color: #FFF3E0;
        border: 2px solid #FF9800;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .diagnosis-header {
        display: flex; align-items: center; gap: 10px;
        color: #E65100 !important; font-weight: 700; font-size: 1.2rem;
    }
    .disease-name {
        color: #1B5E20 !important; font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;
    }
    
    /* Recommendations (Green) */
    .recommendation-box {
        background-color: #E8F5E9;
        border-left: 6px solid #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .rec-header {
        display: flex; align-items: center; gap: 10px;
        color: #2E7D32 !important; font-weight: 700; font-size: 1.4rem; margin-bottom: 1rem;
    }
    
    /* Text inside boxes */
    .recommendation-box p, .recommendation-box li, .diagnosis-box p {
        color: #333333 !important; font-size: 1.05rem; line-height: 1.6;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div { background-color: #4CAF50; }

    /* ---------------- FOOTER ---------------- */
    .custom-footer {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        color: white !important;
        padding: 3rem 2rem;
        margin-top: 4rem;
        text-align: center;
    }
    .custom-footer h3 { color: white !important; margin-bottom: 1rem; }
    .custom-footer p, .custom-footer li { color: rgba(255,255,255,0.8) !important; font-size: 0.95rem; }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. APPLICATION LOGIC CLASS
# -----------------------------------------------------------------------------

class FarmerAssistantApp:
    def __init__(self):
        # Initialize session state variables safely
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.model_loaded = False
            st.session_state.gemini_ready = False
            st.session_state.model = None
            st.session_state.class_mapping = None
            st.session_state.gemini_helper = None
            st.session_state.audio_processor = None

        # Link local variables to session state
        self.model = st.session_state.model
        self.class_mapping = st.session_state.class_mapping
        self.gemini_helper = st.session_state.gemini_helper
        self.audio_processor = st.session_state.audio_processor

    def initialize_components(self):
        """Lazy load heavy components to ensure app stays responsive"""
        if not st.session_state.initialized:
            # 1. Load TensorFlow Model
            if not st.session_state.model_loaded:
                model_path = 'plant_disease_model.h5'
                mapping_path = 'class_mapping.json'
                
                if os.path.exists(model_path) and os.path.exists(mapping_path):
                    try:
                        self.model = tf.keras.models.load_model(model_path)
                        st.session_state.model = self.model
                        st.session_state.model_loaded = True
                        
                        with open(mapping_path, 'r') as f:
                            self.class_mapping = json.load(f)
                            st.session_state.class_mapping = self.class_mapping
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
                else:
                    # Non-blocking warning for UI demo purposes
                    st.warning("⚠️ Model files not found. Image diagnosis will use simulation or fail.")

            # 2. Initialize Gemini AI
            if not st.session_state.gemini_ready:
                try:
                    self.gemini_helper = GeminiHelper()
                    st.session_state.gemini_helper = self.gemini_helper
                    st.session_state.gemini_ready = True
                except Exception as e:
                    st.warning(f"⚠️ Gemini AI could not start. Please check API Key. ({str(e)})")

            # 3. Mark as initialized
            st.session_state.initialized = True

    def process_audio(self, audio_bytes):
        """Helper to process audio bytes via Whisper"""
        if not self.audio_processor:
            try:
                self.audio_processor = AudioProcessor(model_size='base')
                st.session_state.audio_processor = self.audio_processor
            except Exception as e:
                return f"Error initializing Audio Processor: {str(e)}"
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Transcribe
            text = self.audio_processor.process_audio_file(temp_path, use_whisper=True)
            return text
        except Exception as e:
            return f"Error transcribing: {str(e)}"
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    # -------------------------------------------------------------------------
    # UI RENDER FUNCTIONS
    # -------------------------------------------------------------------------

    def render_header_hero(self):
        """Renders the HTML Header and Hero Section"""
        st.markdown("""
        <div class="custom-header">
            <i class="fas fa-leaf"></i>
            <h1>Smart Farmer Assistant</h1>
        </div>
        
        <div class="hero-section">
            <h2 class="hero-title">AI-Powered Agricultural Assistance</h2>
            <p class="hero-subtitle">
                Identify plant diseases, get expert advice, and improve your crop health with cutting-edge AI technology.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Cards in 3 Columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <i class="fas fa-robot feature-icon"></i>
                <h3>AI Disease Detection</h3>
                <p>Advanced CNN model with high accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-card">
                <i class="fas fa-language feature-icon"></i>
                <h3>Multi-Language</h3>
                <p>Supports 90+ languages including Hindi</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-card">
                <i class="fas fa-bolt feature-icon"></i>
                <h3>Instant Results</h3>
                <p>Get recommendations in seconds</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

    def render_text_tab(self):
        """Renders the Text Search Tab"""
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center;">
            <i class="fas fa-search section-header-icon"></i>
            <h2 class="section-title">Text-Based Search</h2>
            <p class="section-subtitle">Describe symptoms or ask questions</p>
        </div>
        """, unsafe_allow_html=True)

        # The text area background is now White with Black text due to CSS above
        query = st.text_area(
            "Your Question:", 
            placeholder="Example: My tomato plants have yellow leaves with brown spots. What should I do?", 
            height=150
        )
        
        # Centered Button Layout
        col_c, col_btn, col_d = st.columns([1, 1, 1])
        with col_btn:
            search_btn = st.button("🔍 Get Expert Solution", type="primary")

        if search_btn:
            if not query:
                st.warning("⚠️ Please enter a question.")
            elif not self.gemini_helper:
                st.error("❌ Gemini AI not initialized.")
            else:
                with st.spinner("🤖 AI is analyzing your query..."):
                    response = self.gemini_helper.get_text_response(query)
                
                # Render Response Box
                st.markdown("""
                <div class="recommendation-box">
                    <div class="rec-header">
                        <i class="fas fa-lightbulb"></i>
                        <span>Agricultural Advice</span>
                    </div>
                """, unsafe_allow_html=True)
                st.write(response)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    def render_image_tab(self):
        """Renders the Image Diagnosis Tab with full prediction logic"""
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center;">
            <i class="fas fa-camera section-header-icon"></i>
            <h2 class="section-title">Image Diagnosis</h2>
            <p class="section-subtitle">Upload a photo for instant detection</p>
        </div>
        """, unsafe_allow_html=True)

        file = st.file_uploader("Upload Plant Image", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

        if file:
            col1, col2 = st.columns(2)
            image = Image.open(file)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("🔬 Analyze Image"):
                    if not self.model:
                        st.error("❌ Model not loaded.")
                    else:
                        with st.spinner("🔬 Analyzing image structure..."):
                            try:
                                # Image Preprocessing
                                img_array = np.array(image.resize((224, 224)))
                                img_array = np.expand_dims(img_array, axis=0)
                                img_array = preprocess_input(img_array)
                                
                                # Prediction
                                predictions = self.model.predict(img_array, verbose=0)
                                idx = np.argmax(predictions[0])
                                confidence = predictions[0][idx]
                                class_name = self.class_mapping['index_to_class'][str(idx)]
                                
                                # Top 3 Predictions Logic
                                top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                                top_3_predictions = [
                                    {
                                        'class': self.class_mapping['index_to_class'][str(i)],
                                        'confidence': float(predictions[0][i])
                                    }
                                    for i in top_3_idx
                                ]
                                
                                # 1. Diagnosis Results Box (Orange)
                                st.markdown(f"""
                                <div class="diagnosis-box">
                                    <div class="diagnosis-header">
                                        <i class="fas fa-bullseye"></i>
                                        <span>Diagnosis Results</span>
                                    </div>
                                    <div class="disease-name">{class_name}</div>
                                    <p style="margin-bottom:5px;">Confidence: <b>{confidence*100:.2f}%</b></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.progress(float(confidence))
                                
                                # Top 3 details
                                with st.expander("📊 View Top 3 Predictions"):
                                    for p in top_3_predictions:
                                        st.write(f"- **{p['class']}**: {p['confidence']*100:.2f}%")
                                
                                # Get Gemini Recommendations
                                with st.spinner("💊 Generating treatment plan..."):
                                    advice = self.gemini_helper.get_image_diagnosis_response(class_name, confidence)
                                
                                # 2. Treatment Box (Green)
                                st.markdown("""
                                <div class="recommendation-box">
                                    <div class="rec-header">
                                        <i class="fas fa-pills"></i>
                                        <span>Treatment Recommendations</span>
                                    </div>
                                """, unsafe_allow_html=True)
                                st.write(advice)
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

    def render_audio_tab(self):
        """Renders the Audio Assistant Tab"""
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center;">
            <i class="fas fa-microphone section-header-icon"></i>
            <h2 class="section-title">Voice-Activated Assistance</h2>
            <p class="section-subtitle">Speak your question in any language</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sub-mode selection using Radio Buttons
        mode = st.radio("Select Input Mode:", ["Record Audio", "Upload Audio File"], horizontal=True, label_visibility="collapsed")
        
        st.markdown("---")

        if mode == "Record Audio":
            # -------------------------------------------------------------
            # CRITICAL FIX: CENTERING THE RECORDER
            # Using columns [10, 2, 10] squeezes the component into the 
            # center, hiding the black rectangle background overlap.
            # -------------------------------------------------------------
            col_left, col_center, col_right = st.columns([10, 2, 10])
            
            with col_center:
                # The Audio Recorder Component
                audio_bytes = audio_recorder(
                    text="",
                    recording_color="#e74c3c",
                    neutral_color="#4CAF50",
                    icon_name="microphone",
                    icon_size="4x",
                )
            
            st.markdown("<p style='text-align:center; color:#888; margin-top:10px;'>Click microphone to record</p>", unsafe_allow_html=True)

            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
                with col_c2:
                    if st.button("📝 Transcribe & Solve", type="primary"):
                        with st.spinner("Processing audio..."):
                            text = self.process_audio(audio_bytes)
                        
                        st.success("Transcription Complete!")
                        st.info(f"🗣️ You asked: {text}")
                        
                        with st.spinner("🤖 Getting expert advice..."):
                            response = self.gemini_helper.get_audio_response(text)
                            
                        st.markdown("""
                        <div class="recommendation-box">
                            <div class="rec-header">
                                <i class="fas fa-lightbulb"></i>
                                <span>Agricultural Advice</span>
                            </div>
                        """, unsafe_allow_html=True)
                        st.write(response)
                        st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Upload Mode
            uploaded_audio = st.file_uploader("Upload audio file (WAV/MP3)", type=['wav', 'mp3'])
            if uploaded_audio:
                col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
                with col_c2:
                    if st.button("Analyze Audio File", type="primary"):
                        with st.spinner("Processing file..."):
                            text = self.process_audio(uploaded_audio.read())
                            
                        st.markdown(f"**🗣️ Question:** {text}")
                        
                        with st.spinner("🤖 Getting expert advice..."):
                            response = self.gemini_helper.get_audio_response(text)
                            
                        st.markdown("""
                        <div class="recommendation-box">
                            <div class="rec-header">
                                <i class="fas fa-lightbulb"></i>
                                <span>Agricultural Advice</span>
                            </div>
                        """, unsafe_allow_html=True)
                        st.write(response)
                        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    def render_footer(self):
        """Renders the Footer"""
        st.markdown("""
        <footer class="custom-footer">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; max-width: 1000px; margin: 0 auto; text-align: left;">
                <div>
                    <h3>About</h3>
                    <p>AI-powered agricultural assistance designed to help modern farmers identify diseases and improve crop health.</p>
                </div>
                <div>
                    <h3>Features</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li>🌱 Plant Disease Detection</li>
                        <li>👨‍⚕️ Expert Advice</li>
                        <li>🗣️ Multi-Language Support</li>
                    </ul>
                </div>
                <div>
                    <h3>Tech Stack</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li>TensorFlow / Keras</li>
                        <li>Google Gemini AI</li>
                        <li>Whisper STT</li>
                    </ul>
                </div>
            </div>
            <div style="margin-top: 3rem; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1.5rem;">
                <p>Made with ❤️ for Farmers Worldwide 🌾</p>
            </div>
        </footer>
        """, unsafe_allow_html=True)

    def run(self):
        """Main execution flow"""
        self.initialize_components()
        self.render_header_hero()
        
        # Navigation Tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Text Search", "📸 Image Diagnosis", "🎤 Audio Help"])
        
        with tab1:
            self.render_text_tab()
        with tab2:
            self.render_image_tab()
        with tab3:
            self.render_audio_tab()
            
        self.render_footer()

if __name__ == "__main__":
    app = FarmerAssistantApp()
    app.run()