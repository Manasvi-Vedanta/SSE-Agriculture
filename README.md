# 🌾 Smart Search Engine for Farmers

An intelligent, multi-modal agricultural assistance system powered by Deep Learning and AI. This application helps farmers identify plant diseases and receive expert agricultural advice through text, image, and voice inputs.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)

## 🌟 Features

### 🔍 **Text Search**
- Natural language query interface
- Instant agricultural advice powered by Google Gemini AI
- Detailed solutions with causes, symptoms, treatments, and prevention measures

### 📸 **Image-Based Diagnosis**
- Upload plant images for disease detection
- CNN model with MobileNetV2 architecture (54.39% validation accuracy)
- Confidence scores and top-3 predictions
- Low confidence warnings for uncertain predictions (<60%)
- Comprehensive disease information from AI

### 🎤 **Voice Input**
- **Real-time audio recording** with one-click interface
- Multi-language support (English, Hindi, and 90+ languages)
- Automatic language detection using OpenAI Whisper
- Speech-to-text transcription with AI-powered responses

## 🎯 Key Capabilities

- **28 Plant Disease Classes**: Covers multiple crops including Apple, Corn, Grape, Tomato, and more
- **Multi-Modal Input**: Accept text, image, and audio queries
- **Intelligent AI Responses**: 300-500 word detailed explanations with structured sections
- **Transfer Learning**: Fine-tuned MobileNetV2 for efficient disease classification
- **Language Flexibility**: Automatic Hindi/English detection and transcription

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Web UI                       │
├─────────────┬─────────────────┬─────────────────────────┤
│  Text Input │  Image Upload   │  Audio Recording        │
└──────┬──────┴────────┬────────┴──────────┬──────────────┘
       │               │                   │
       ▼               ▼                   ▼
  ┌────────┐    ┌──────────────┐    ┌────────────┐
  │ Gemini │    │ CNN Model    │    │  Whisper   │
  │   AI   │    │ (MobileNetV2)│    │   STT      │
  └────┬───┘    └──────┬───────┘    └─────┬──────┘
       │               │                    │
       └───────────────┴────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Gemini AI      │
              │  (Final Response)│
              └─────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.12 |
| **Web Framework** | Streamlit 1.30+ |
| **Deep Learning** | TensorFlow 2.17+, Keras |
| **Transfer Learning** | MobileNetV2 (ImageNet pretrained) |
| **LLM** | Google Gemini 2.5-flash |
| **Speech-to-Text** | OpenAI Whisper (base model) |
| **Audio Processing** | SoundFile, SoundDevice, SpeechRecognition |
| **Audio Recording** | audio-recorder-streamlit |
| **Data Processing** | NumPy, Pandas, PIL |
| **Visualization** | Matplotlib, Seaborn |

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- Git (for cloning the dataset)
- WSL (for Windows users to handle long file paths)
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd smart-farmer-search-engine
```

### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\venv\Scripts\activate.bat

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Clone the Dataset

The project uses the [PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) for training.

**For Windows users** (to handle long file paths):
```bash
# Using WSL
wsl
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
exit
```

**For Linux/Mac users**:
```bash
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
```

**Enable Windows Long Path Support** (if needed):
```powershell
# Run as Administrator
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
# Restart your system for changes to take effect
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 6: Train the Model (Required)

Train the CNN model on the PlantDoc-Dataset:

```bash
python train_model.py
```

**Training Details:**
- Phase 1: 30 epochs with frozen MobileNetV2 base
- Phase 2: 10 epochs with fine-tuning (last 30 layers unfrozen)
- Validation accuracy: ~54.39%
- Training time: ~45-60 minutes (depending on hardware)
- Generates: `plant_disease_model.h5` (19.1 MB), `class_mapping.json`, `training_history.png`

**Note:** Pre-trained model files are not included in this repository due to size constraints. You must train the model yourself.

## 🚀 Usage

### Launch the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

#### 1️⃣ **Text Search Tab**
1. Type your agricultural question or describe plant symptoms
2. Click "Get Advice" 
3. Receive detailed AI-generated guidance with:
   - Overview of the issue
   - Main causes
   - Symptoms to look for
   - Treatment steps (4-5 detailed steps)
   - Prevention measures (3-4 strategies)
   - Additional care tips

#### 2️⃣ **Image Diagnosis Tab**
1. Upload a plant image (JPG, JPEG, PNG)
2. Click "Analyze Image"
3. View:
   - Disease prediction with confidence score
   - Top-3 alternative predictions
   - Confidence bar visualization
   - Comprehensive treatment recommendations from Gemini AI
   - Low confidence warning (if <60%)

#### 3️⃣ **Audio Help Tab**

**Record Audio:**
1. Click the microphone button to start recording
2. Speak your question in English, Hindi, or any supported language
3. Click the button again to stop recording
4. Audio playback appears automatically
5. Click "Transcribe & Get Solution"
6. View:
   - Detected language
   - Transcribed text
   - AI-generated agricultural advice

**Type Question:**
1. Switch to "Type Question" tab
2. Enter your question manually
3. Click "Get Answer"
4. Receive AI-generated advice

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Architecture** | MobileNetV2 + Custom Classification Head |
| **Total Parameters** | 3,052,380 |
| **Trainable Parameters** | 794,396 |
| **Training Samples** | 1,869 images |
| **Validation Samples** | 467 images |
| **Disease Classes** | 28 |
| **Best Validation Accuracy** | 54.39% |
| **Model Size** | 19.1 MB |
| **Input Size** | 224x224 RGB |

### Model Architecture Details

```
MobileNetV2 (ImageNet weights, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, ReLU) + Dropout(0.5)
    ↓
Dense(256, ReLU) + Dropout(0.3)
    ↓
Dense(28, Softmax)
```

**Note:** Accuracy is limited by the relatively small dataset (~67 images per class). The system includes confidence warnings (<60%) to alert users when predictions are uncertain.

## 🌿 Supported Plant Diseases (28 Classes)

The model can identify diseases across multiple crops:

- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Cherry**: Powdery mildew, Healthy
- **Corn**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca, Leaf blight, Healthy
- **Peach**: Bacterial spot, Healthy
- **Pepper**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Strawberry**: Leaf scorch, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Mosaic virus, Yellow Leaf Curl Virus, Healthy

## 🗂️ Project Structure

```
smart-farmer-search-engine/
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── data_loader_v2.py           # Data loading with long path support
├── gemini_helper.py            # Google Gemini API integration
├── audio_processor.py          # Audio recording & Whisper STT
├── predict.py                  # Standalone prediction script
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── class_mapping.json          # Generated: Class index to name mapping
├── plant_disease_model.h5      # Generated: Trained model (19.1 MB)
├── training_history.png        # Generated: Training plots
│
└── PlantDoc-Dataset/          # Clone separately (not in repo)
    ├── train/                 # Training images (1,869 images)
    └── test/                  # Testing images (236 images)
```

## 🔧 Configuration

### Gemini AI Settings

Edit `gemini_helper.py` to customize AI behavior:

```python
generation_config = {
    "temperature": 0.7,           # Creativity (0.0-1.0)
    "top_p": 0.9,                 # Nucleus sampling
    "top_k": 40,                  # Top-k sampling
    "max_output_tokens": 1024,    # Maximum response length
}
```

### Whisper Model Size

Edit `audio_processor.py` to change Whisper model:

```python
# Options: 'tiny', 'base', 'small', 'medium', 'large'
# 'base' is recommended for balance between speed and accuracy
AudioProcessor(model_size='base')
```

| Model | Parameters | Speed | Accuracy |
|-------|-----------|-------|----------|
| tiny | 39M | ~32x | Good |
| base | 74M | ~16x | Better |
| small | 244M | ~6x | Great |
| medium | 769M | ~2x | Excellent |
| large | 1550M | ~1x | Best |

### Model Training Parameters

Edit `train_model.py` to adjust:

```python
IMG_SIZE = (224, 224)           # Image dimensions
BATCH_SIZE = 32                 # Batch size
EPOCHS_PHASE1 = 30              # Initial training epochs
EPOCHS_PHASE2 = 10              # Fine-tuning epochs
LEARNING_RATE = 0.001           # Initial learning rate
FINE_TUNE_LEARNING_RATE = 0.0001  # Fine-tuning rate
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

### Areas for Contribution

- Expand dataset with more plant diseases
- Improve model accuracy
- Add more language support for UI
- Implement real-time camera feed
- Add mobile-responsive design
- Create Docker containerization
- Add unit tests
- Improve documentation

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **PlantDoc Dataset**: [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- **MobileNetV2**: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- **OpenAI Whisper**: [Radford et al., 2022](https://github.com/openai/whisper)
- **Google Gemini**: [Google DeepMind](https://deepmind.google/technologies/gemini/)
- **Streamlit**: [Streamlit Inc.](https://streamlit.io/)
- **TensorFlow**: [Google Brain Team](https://www.tensorflow.org/)

## 🚨 Known Limitations

1. **Model Accuracy**: ~54% validation accuracy due to limited training data (~67 images per class)
2. **Test Set Mismatch**: Test set has 27 classes vs 28 in training set
3. **File Size**: Model file (19.1 MB) must be generated locally
4. **API Costs**: Google Gemini API usage may incur costs
5. **Windows Long Paths**: Requires WSL for dataset cloning on Windows
6. **Language Support**: Gemini responses primarily in English (multilingual input supported)

## 🔮 Future Improvements

- [ ] Increase dataset size for better accuracy (target: 500+ images per class)
- [ ] Add more plant disease classes (target: 50+ diseases)
- [ ] Implement real-time camera feed for diagnosis
- [ ] Add multi-language UI support (Hindi, Spanish, French, etc.)
- [ ] Deploy as web service (Streamlit Cloud/AWS/Azure)
- [ ] Add user feedback mechanism for predictions
- [ ] Implement model ensemble for better accuracy
- [ ] Add geolocation-based disease prevalence data
- [ ] Create mobile application (React Native/Flutter)
- [ ] Add crop management calendar
- [ ] Integrate weather API for context-aware recommendations

## 🐛 Troubleshooting

### Model Not Found Error
```
Solution: Run `python train_model.py` to train the model first
```

### Gemini API Error
```
Solution: 
1. Verify API key in .env file
2. Check internet connection
3. Verify API quota limits
```

### Audio Recording Not Working
```
Solution:
1. Grant microphone permissions in browser
2. Use HTTPS or localhost (browser security requirement)
3. Check browser compatibility (Chrome/Edge recommended)
```

### Long Path Errors (Windows)
```
Solution:
1. Use WSL for git clone operations
2. Enable Windows long path support (see Installation Step 4)
3. Restart system after enabling
```

---

**Made with ❤️ for farmers worldwide (i definitely did not create the readme file using AI ~ Manasvi)** 🌾

*Empowering agriculture with AI technology*
