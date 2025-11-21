"""
Smart Search Engine for Farmers - Streamlit Application
Multi-modal agricultural assistance with Text, Image, and Audio inputs
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
import tempfile
from audio_recorder_streamlit import audio_recorder

# Import custom modules
from gemini_helper import GeminiHelper
from audio_processor import AudioProcessor
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page configuration
st.set_page_config(
    page_title="Smart Search Engine for Farmers",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #FFF3E0;
        border: 2px solid #FF9800;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)


class FarmerAssistantApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        # Initialize session state first - always ensure these keys exist
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'gemini_ready' not in st.session_state:
            st.session_state.gemini_ready = False
        if 'audio_ready' not in st.session_state:
            st.session_state.audio_ready = False
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'class_mapping' not in st.session_state:
            st.session_state.class_mapping = None
        if 'gemini_helper' not in st.session_state:
            st.session_state.gemini_helper = None
        if 'audio_processor' not in st.session_state:
            st.session_state.audio_processor = None
        
        # Use session state to persist objects
        self.model = st.session_state.model
        self.class_mapping = st.session_state.class_mapping
        self.gemini_helper = st.session_state.gemini_helper
        self.audio_processor = st.session_state.audio_processor
    
    def load_model(self):
        """Load the trained plant disease model"""
        if not st.session_state.model_loaded:
            model_path = 'plant_disease_model.h5'
            
            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found: {model_path}")
                st.info("📝 Please train the model first by running: `python train_model.py`")
                return False
            
            try:
                with st.spinner("🔄 Loading plant disease detection model..."):
                    self.model = tf.keras.models.load_model(model_path)
                    st.session_state.model = self.model
                    st.session_state.model_loaded = True
                return True
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                return False
        else:
            # Model already loaded, just return True
            return True
    
    def load_class_mapping(self):
        """Load class name mappings"""
        mapping_path = 'class_mapping.json'
        
        if not os.path.exists(mapping_path):
            st.error(f"❌ Class mapping file not found: {mapping_path}")
            st.info("📝 Please train the model first to generate the class mapping.")
            return False
        
        try:
            with open(mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
                st.session_state.class_mapping = self.class_mapping
            return True
        except Exception as e:
            st.error(f"❌ Error loading class mapping: {str(e)}")
            return False
    
    def initialize_gemini(self):
        """Initialize Gemini helper"""
        if not st.session_state.gemini_ready:
            try:
                with st.spinner("🤖 Initializing Gemini AI..."):
                    self.gemini_helper = GeminiHelper()
                    st.session_state.gemini_helper = self.gemini_helper
                st.session_state.gemini_ready = True
                return True
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                st.info("📝 Please create a .env file with your GEMINI_API_KEY")
                return False
            except Exception as e:
                st.error(f"❌ Error initializing Gemini: {str(e)}")
                return False
        return True
    
    def initialize_audio_processor(self):
        """Initialize audio processor"""
        if not st.session_state.audio_ready:
            try:
                self.audio_processor = AudioProcessor(model_size='base')
                st.session_state.audio_processor = self.audio_processor
                st.session_state.audio_ready = True
                return True
            except Exception as e:
                st.error(f"❌ Error initializing audio processor: {str(e)}")
                return False
        return True
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess image for model prediction"""
        # Resize image
        image = image.resize(target_size)
        
        # Convert to array
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def predict_disease(self, image):
        """Predict plant disease from image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Get class name
            predicted_class = self.class_mapping['index_to_class'][str(predicted_class_idx)]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'class': self.class_mapping['index_to_class'][str(idx)],
                    'confidence': float(predictions[0][idx])
                }
                for idx in top_3_idx
            ]
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'top_3': top_3_predictions
            }
        
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            return None
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🌾 Smart Search Engine for Farmers</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Welcome!</b> This AI-powered assistant helps you identify plant diseases and get agricultural advice through:
        <ul>
            <li>🔍 <b>Text Search</b> - Describe symptoms and get solutions</li>
            <li>📸 <b>Image Diagnosis</b> - Upload plant images for disease detection</li>
            <li>🎤 <b>Audio Help</b> - Speak your concerns and get assistance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with information"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/plant.png", width=100)
            st.title("About")
            
            st.markdown("""
            ### 🌱 Features
            - **AI-Powered Disease Detection**
            - **Multi-Modal Input Support**
            - **Expert Agricultural Advice**
            - **Instant Recommendations**
            
            ### 📊 Model Info
            """)
            
            if st.session_state.model_loaded and self.class_mapping:
                st.success("✅ Model Loaded")
                st.info(f"Classes: {self.class_mapping['num_classes']}")
            else:
                st.warning("⚠️ Model Not Loaded")
            
            if st.session_state.gemini_ready:
                st.success("✅ Gemini AI Ready")
            else:
                st.warning("⚠️ Gemini Not Initialized")
            
            st.markdown("---")
            st.markdown("""
            ### 🔧 Tech Stack
            - TensorFlow/Keras
            - Google Gemini
            - Streamlit
            - Whisper (Audio)
            """)
            
            st.markdown("---")
            st.markdown("Made with ❤️ for Farmers")
    
    def render_text_search_tab(self):
        """Render text search interface"""
        st.markdown('<h2 class="sub-header">🔍 Text-Based Search</h2>', unsafe_allow_html=True)
        
        st.write("Describe your plant's symptoms or ask any agricultural question:")
        
        # Text input
        user_query = st.text_area(
            "Your Question:",
            placeholder="Example: My tomato plants have yellow leaves with brown spots. What should I do?",
            height=100
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_button = st.button("🔍 Get Solution", type="primary", use_container_width=True)
        
        if search_button:
            if not user_query.strip():
                st.warning("⚠️ Please enter your question or description.")
                return
            
            if not self.gemini_helper:
                st.error("❌ Gemini AI is not initialized. Please check your API key.")
                return
            
            with st.spinner("🤔 Analyzing your query..."):
                response = self.gemini_helper.get_text_response(user_query)
            
            st.markdown("### 📝 Agricultural Advice")
            st.markdown(response)
    
    def render_image_diagnosis_tab(self):
        """Render image diagnosis interface"""
        st.markdown('<h2 class="sub-header">📸 Image-Based Diagnosis</h2>', unsafe_allow_html=True)
        
        st.write("Upload a photo of your plant for disease detection:")
        
        uploaded_file = st.file_uploader(
            "Choose a plant image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of the affected plant"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.markdown("### 🔬 Analysis")
                
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    if not self.model or not self.class_mapping:
                        st.error("❌ Model not loaded. Please check if the model file exists.")
                        return
                    
                    with st.spinner("🔬 Analyzing image..."):
                        # Predict disease
                        result = self.predict_disease(image)
                    
                    if result:
                        # Display prediction
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown(f"### 🎯 Prediction Results")
                        st.markdown(f"**Detected Condition:** {result['predicted_class']}")
                        st.markdown(f"**Confidence:** {result['confidence']*100:.2f}%")
                        
                        # Show confidence bar
                        st.progress(result['confidence'])
                        
                        st.markdown("#### 📊 Top 3 Predictions:")
                        for i, pred in enumerate(result['top_3'], 1):
                            st.write(f"{i}. {pred['class']} - {pred['confidence']*100:.2f}%")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Get Gemini recommendation
                        st.markdown("---")
                        st.markdown("### 💡 Treatment Recommendations")
                        
                        with st.spinner("🤖 Generating treatment recommendations..."):
                            gemini_response = self.gemini_helper.get_image_diagnosis_response(
                                result['predicted_class'],
                                result['confidence']
                            )
                        
                        st.markdown(gemini_response)
    
    def render_audio_help_tab(self):
        """Render audio help interface"""
        st.markdown('<h2 class="sub-header">🎤 Audio-Based Help</h2>', unsafe_allow_html=True)
        
        st.write("Record your question or type it manually:")
        
        # Create tabs for recording vs manual input
        audio_tab1, audio_tab2 = st.tabs(["🎙️ Record Audio", "📝 Type Question"])
        
        with audio_tab1:
            st.info("💡 Click the microphone button below to start recording. Click again to stop.")
            
            # Audio recorder widget
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="3x",
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format='audio/wav')
                
                if st.button("🎯 Transcribe & Get Solution", type="primary"):
                    if not self.audio_processor:
                        self.initialize_audio_processor()
                    
                    # Save recorded audio to temp file
                    temp_audio_path = None
                    try:
                        with st.spinner("💾 Processing audio..."):
                            # Create temp file with proper context manager
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as temp_file:
                                temp_file.write(audio_bytes)
                                temp_audio_path = temp_file.name
                            
                            # Verify file exists
                            if not os.path.exists(temp_audio_path):
                                raise FileNotFoundError(f"Failed to create temporary audio file")
                            
                            # Transcribe
                            with st.spinner("🎤 Transcribing audio..."):
                                transcription = self.audio_processor.process_audio_file(temp_audio_path, use_whisper=True)
                                
                                st.success("✅ Transcription Complete!")
                                st.markdown("### 📝 Transcribed Text:")
                                st.info(transcription)
                                
                                # Get Gemini response
                                st.markdown("### 💡 Agricultural Advice:")
                                with st.spinner("🤖 Generating response..."):
                                    response = self.gemini_helper.get_audio_response(transcription)
                                
                                st.markdown(response)
                                
                    except Exception as e:
                        st.error(f"❌ Error processing audio: {str(e)}")
                        import traceback
                        st.error(f"Details: {traceback.format_exc()}")
                    finally:
                        # Clean up temp file
                        if temp_audio_path and os.path.exists(temp_audio_path):
                            try:
                                os.unlink(temp_audio_path)
                            except:
                                pass
        
        with audio_tab2:
            st.info("💡 If you prefer, you can type what you want to ask")
            
            manual_text = st.text_area(
                "Type your question:",
                placeholder="Example: How do I treat rust on wheat leaves?",
                height=100
            )
            
            if st.button("🔍 Get Answer", type="primary"):
                if not manual_text.strip():
                    st.warning("⚠️ Please enter your question.")
                else:
                    with st.spinner("🤖 Generating response..."):
                        response = self.gemini_helper.get_text_response(manual_text)
                    
                    st.markdown("### 💡 Agricultural Advice:")
                    st.markdown(response)
    
    def run(self):
        """Main application entry point"""
        # Render header and sidebar
        self.render_header()
        self.render_sidebar()
        
        # Initialize components
        if not st.session_state.initialized:
            with st.spinner("🚀 Initializing application..."):
                # Load model and mappings
                model_ok = self.load_model()
                mapping_ok = self.load_class_mapping() if model_ok else False
                
                # Initialize Gemini
                gemini_ok = self.initialize_gemini()
                
                if model_ok and mapping_ok and gemini_ok:
                    st.session_state.initialized = True
                    st.success("✅ Application initialized successfully!")
                else:
                    st.error("❌ Failed to initialize some components. Please check the errors above.")
                    st.stop()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Text Search", "📸 Image Diagnosis", "🎤 Audio Help"])
        
        with tab1:
            self.render_text_search_tab()
        
        with tab2:
            self.render_image_diagnosis_tab()
        
        with tab3:
            self.render_audio_help_tab()


def main():
    """Main function to run the Streamlit app"""
    app = FarmerAssistantApp()
    app.run()


if __name__ == "__main__":
    main()
