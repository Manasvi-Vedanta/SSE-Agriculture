"""
Audio Processing Module for Speech-to-Text
Supports both audio recording and file upload with Whisper transcription
"""

import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import speech_recognition as sr
from pathlib import Path
import tempfile
import warnings
import io
warnings.filterwarnings('ignore')


class AudioProcessor:
    """Handles audio recording and transcription"""
    
    def __init__(self, model_size='base'):
        """
        Initialize audio processor
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.whisper_model = None
        self.sample_rate = 16000  # Whisper uses 16kHz
        
    def load_whisper_model(self):
        """Load Whisper model for transcription"""
        if self.whisper_model is None:
            print(f"Loading Whisper {self.model_size} model...")
            self.whisper_model = whisper.load_model(self.model_size)
            print("Whisper model loaded successfully!")
        return self.whisper_model
    
    def record_audio(self, duration=5, filename="recorded_audio.wav"):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            filename: Output filename for the recording
            
        Returns:
            Path to the saved audio file
        """
        print(f"Recording for {duration} seconds...")
        print("Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        # Save audio file
        sf.write(filename, audio_data, self.sample_rate)
        print(f"Recording saved to {filename}")
        
        return filename
    
    def transcribe_with_whisper(self, audio_path, language=None):
        """
        Transcribe audio using Whisper with automatic language detection
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'hi'). If None, auto-detects language
            
        Returns:
            Transcribed text with detected language
        """
        # Load model if not already loaded
        if self.whisper_model is None:
            self.load_whisper_model()
        
        print("Transcribing audio with Whisper...")
        
        try:
            # Load audio using soundfile (doesn't require ffmpeg)
            audio_data, sample_rate = sf.read(audio_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Ensure float32 dtype
            audio_data = audio_data.astype(np.float32)
            
            # Resample to 16kHz if needed (Whisper's expected sample rate)
            if sample_rate != 16000:
                # Simple resampling with float32 dtype
                duration = len(audio_data) / sample_rate
                target_length = int(duration * 16000)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), target_length, dtype=np.float32),
                    np.arange(len(audio_data), dtype=np.float32),
                    audio_data
                ).astype(np.float32)
            
            # Transcribe with automatic language detection or specified language
            if language:
                result = self.whisper_model.transcribe(
                    audio_data,
                    language=language,
                    task='transcribe'
                )
                detected_language = language
            else:
                # Auto-detect language (supports Hindi, English, and 90+ other languages)
                result = self.whisper_model.transcribe(audio_data)
                detected_language = result.get('language', 'unknown')
            
            transcription = result['text'].strip()
            print(f"Detected Language: {detected_language}")
            print(f"Transcription: {transcription}")
            
            return transcription
            
        except Exception as e:
            print(f"Error in Whisper transcription: {str(e)}")
            raise
    
    def transcribe_with_speech_recognition(self, audio_path):
        """
        Transcribe audio using speech_recognition library (fallback method)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
                
            # Try Google Speech Recognition
            print("Transcribing with Google Speech Recognition...")
            text = recognizer.recognize_google(audio)
            print(f"Transcription: {text}")
            return text
            
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"
        except Exception as e:
            return f"Error during transcription: {e}"
    
    def process_audio_file(self, audio_path, use_whisper=True):
        """
        Process an audio file and return transcription
        
        Args:
            audio_path: Path to audio file
            use_whisper: Whether to use Whisper (True) or speech_recognition (False)
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if use_whisper:
            return self.transcribe_with_whisper(audio_path)
        else:
            return self.transcribe_with_speech_recognition(audio_path)
    
    def save_uploaded_audio(self, uploaded_file):
        """
        Save uploaded audio file from Streamlit
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Path to saved file
        """
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    @staticmethod
    def get_audio_info(audio_path):
        """Get information about an audio file"""
        try:
            data, sample_rate = sf.read(audio_path)
            duration = len(data) / sample_rate
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': data.shape[1] if len(data.shape) > 1 else 1,
                'samples': len(data)
            }
        except Exception as e:
            return {'error': str(e)}


class StreamlitAudioRecorder:
    """Wrapper for Streamlit audio recording functionality"""
    
    @staticmethod
    def record_audio_streamlit(duration=5):
        """
        Record audio in Streamlit app
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Path to recorded audio file
        """
        processor = AudioProcessor()
        
        # Generate unique filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp_audio/recording_{timestamp}.wav"
        
        # Ensure directory exists
        Path("temp_audio").mkdir(exist_ok=True)
        
        # Record
        audio_path = processor.record_audio(duration, filename)
        
        return audio_path


def test_audio_processor():
    """Test the audio processor"""
    print("=" * 60)
    print("Audio Processor Test")
    print("=" * 60)
    
    processor = AudioProcessor(model_size='base')
    
    # Test recording
    print("\nTest 1: Recording audio...")
    print("You will have 5 seconds to speak after 'Speak now!' appears")
    input("Press Enter to start recording...")
    
    audio_file = processor.record_audio(duration=5, filename="test_recording.wav")
    
    # Get audio info
    info = processor.get_audio_info(audio_file)
    print(f"\nAudio Info: {info}")
    
    # Test transcription with Whisper
    print("\nTest 2: Transcribing with Whisper...")
    transcription = processor.transcribe_with_whisper(audio_file)
    print(f"Result: {transcription}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_audio_processor()
