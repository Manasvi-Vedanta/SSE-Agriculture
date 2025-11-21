"""
Quick Start Script - Setup and Run Smart Farmer Assistant
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ {description} found")
        return True
    else:
        print(f"❌ {description} NOT found: {filepath}")
        return False

def check_environment():
    """Check if environment is set up correctly"""
    print_header("Checking Environment")
    
    checks = {
        'Dataset': check_file_exists('PlantDoc-Dataset', 'PlantDoc-Dataset folder'),
        'Requirements': check_file_exists('requirements.txt', 'requirements.txt'),
        'App': check_file_exists('app.py', 'app.py'),
        'Training Script': check_file_exists('train_model.py', 'train_model.py'),
    }
    
    # Check for .env file
    env_exists = check_file_exists('.env', '.env file')
    if not env_exists:
        print("⚠️  Please create .env file from .env.example and add your GEMINI_API_KEY")
        checks['Env File'] = False
    else:
        checks['Env File'] = True
    
    # Check for trained model
    model_exists = check_file_exists('plant_disease_model.h5', 'Trained model')
    if not model_exists:
        print("⚠️  Model not found. You'll need to train it first.")
        checks['Model'] = False
    else:
        checks['Model'] = True
    
    return all(checks.values())

def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    try:
        print("Installing packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def train_model():
    """Train the model"""
    print_header("Training Model")
    
    response = input("Do you want to train the model now? This may take 30-60 minutes. (y/n): ")
    
    if response.lower() == 'y':
        try:
            print("\n🚀 Starting model training...")
            subprocess.check_call([sys.executable, "train_model.py"])
            print("✅ Model training completed!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Model training failed")
            return False
    else:
        print("⚠️  Skipping model training. You can train later with: python train_model.py")
        return False

def run_app():
    """Run the Streamlit app"""
    print_header("Launching Application")
    
    try:
        print("🚀 Starting Streamlit app...")
        print("The app will open in your browser at http://localhost:8501")
        print("\nPress Ctrl+C to stop the app\n")
        subprocess.check_call(["streamlit", "run", "app.py"])
    except subprocess.CalledProcessError:
        print("❌ Failed to start the app")
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")

def main():
    """Main setup flow"""
    print_header("Smart Search Engine for Farmers - Quick Start")
    
    # Step 1: Check environment
    if not check_environment():
        print("\n⚠️  Some required files are missing. Please check the errors above.")
        return
    
    # Step 2: Check if we need to install dependencies
    print_header("Setup Options")
    print("1. Install dependencies")
    print("2. Train model (if not already trained)")
    print("3. Run application")
    print("4. Full setup (All of the above)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1':
        install_dependencies()
    elif choice == '2':
        train_model()
    elif choice == '3':
        run_app()
    elif choice == '4':
        # Full setup
        if install_dependencies():
            if not Path('plant_disease_model.h5').exists():
                train_model()
            run_app()
    else:
        print("👋 Exiting...")
        return
    
    print_header("Setup Complete!")
    print("📚 For detailed instructions, see README.md")
    print("🐛 For troubleshooting, check the README troubleshooting section")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
