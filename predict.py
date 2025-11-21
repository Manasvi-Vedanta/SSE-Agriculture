"""
Model Prediction Utility
Standalone script for testing model predictions on images
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import sys
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class PlantDiseasePredictor:
    """Utility class for making predictions with trained model"""
    
    def __init__(self, model_path='plant_disease_model.h5', mapping_path='class_mapping.json'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            mapping_path: Path to class mapping JSON
        """
        self.model_path = model_path
        self.mapping_path = mapping_path
        self.model = None
        self.class_mapping = None
        
        self.load_model()
        self.load_mapping()
    
    def load_model(self):
        """Load the trained model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully!")
    
    def load_mapping(self):
        """Load class mapping"""
        if not Path(self.mapping_path).exists():
            raise FileNotFoundError(f"Class mapping not found: {self.mapping_path}")
        
        with open(self.mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        print(f"✅ Loaded {self.class_mapping['num_classes']} classes")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize
        image = image.resize(target_size)
        
        # Convert to array
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def predict(self, image_path, top_k=5):
        """
        Make prediction on an image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.class_mapping['index_to_class'][str(idx)]
            confidence = float(predictions[idx])
            results.append({
                'class': class_name,
                'confidence': confidence,
                'confidence_percent': confidence * 100
            })
        
        return {
            'top_prediction': results[0],
            'all_predictions': results
        }
    
    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                results.append({
                    'image': img_path,
                    'prediction': result
                })
            except Exception as e:
                results.append({
                    'image': img_path,
                    'error': str(e)
                })
        
        return results
    
    def print_prediction(self, result):
        """Pretty print prediction result"""
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        top = result['top_prediction']
        print(f"\n🎯 Top Prediction: {top['class']}")
        print(f"📊 Confidence: {top['confidence_percent']:.2f}%")
        print(f"{'█' * int(top['confidence_percent'] / 2)}")
        
        print("\n📋 All Top Predictions:")
        print("-"*60)
        for i, pred in enumerate(result['all_predictions'], 1):
            bar = '█' * int(pred['confidence_percent'] / 5)
            print(f"{i}. {pred['class']:<40} {pred['confidence_percent']:>6.2f}% {bar}")
        
        print("="*60 + "\n")


def main():
    """Main function for CLI usage"""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py sample_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    try:
        # Initialize predictor
        print("Initializing predictor...")
        predictor = PlantDiseasePredictor()
        
        # Make prediction
        print(f"\nAnalyzing image: {image_path}")
        result = predictor.predict(image_path, top_k=5)
        
        # Print results
        predictor.print_prediction(result)
        
        # Save results to JSON (optional)
        output_file = Path(image_path).stem + "_prediction.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        
        print(f"💾 Results saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Please ensure:")
        print("  1. Model file exists (plant_disease_model.h5)")
        print("  2. Class mapping exists (class_mapping.json)")
        print("  3. Run 'python train_model.py' if files are missing")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
