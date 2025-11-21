"""
Data Loader and Preprocessing Pipeline for PlantDoc-Dataset
This module handles loading, preprocessing, and splitting the dataset.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
import json
from pathlib import Path


class PlantDataLoader:
    """Handles loading and preprocessing of plant disease dataset"""
    
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader
        
        Args:
            dataset_path: Path to PlantDoc-Dataset folder
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / 'train'
        self.test_path = self.dataset_path / 'test'
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = None
        
    def get_class_names(self):
        """Extract class names from train folder structure"""
        if self.train_path.exists():
            self.class_names = sorted([d.name for d in self.train_path.iterdir() if d.is_dir()])
            print(f"Found {len(self.class_names)} classes")
            return self.class_names
        else:
            raise FileNotFoundError(f"Train directory not found: {self.train_path}")
    
    def create_data_generators(self, validation_split=0.2):
        """
        Create data generators for training, validation, and testing
        
        Args:
            validation_split: Fraction of training data to use for validation
            
        Returns:
            train_generator, validation_generator, test_generator, class_names
        """
        # Get class names
        self.get_class_names()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only preprocessing for validation and test
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            str(self.train_path),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            str(self.train_path),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Test generator
        test_generator = None
        if self.test_path.exists():
            test_generator = test_datagen.flow_from_directory(
                str(self.test_path),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
        
        # Save class names mapping
        self.save_class_mapping(train_generator.class_indices)
        
        return train_generator, validation_generator, test_generator, self.class_names
    
    def save_class_mapping(self, class_indices):
        """Save class name to index mapping"""
        # Reverse the mapping (index -> class_name)
        index_to_class = {v: k for k, v in class_indices.items()}
        
        mapping = {
            'class_indices': class_indices,
            'index_to_class': index_to_class,
            'num_classes': len(class_indices)
        }
        
        with open('class_mapping.json', 'w') as f:
            json.dump(mapping, f, indent=4)
        
        print(f"Class mapping saved to class_mapping.json")
        return mapping
    
    @staticmethod
    def load_class_mapping():
        """Load class name to index mapping"""
        try:
            with open('class_mapping.json', 'r') as f:
                mapping = json.load(f)
            return mapping
        except FileNotFoundError:
            print("class_mapping.json not found. Please train the model first.")
            return None
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        info = {
            'train_path': str(self.train_path),
            'test_path': str(self.test_path),
            'num_classes': len(self.class_names) if self.class_names else 0,
            'class_names': self.class_names,
            'image_size': self.img_size,
            'batch_size': self.batch_size
        }
        
        # Count images per class in train
        if self.train_path.exists():
            train_counts = {}
            for class_dir in self.train_path.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png')))
                    train_counts[class_dir.name] = count
            info['train_counts'] = train_counts
            info['total_train_images'] = sum(train_counts.values())
        
        # Count images per class in test
        if self.test_path.exists():
            test_counts = {}
            for class_dir in self.test_path.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png')))
                    test_counts[class_dir.name] = count
            info['test_counts'] = test_counts
            info['total_test_images'] = sum(test_counts.values())
        
        return info


if __name__ == "__main__":
    # Test the data loader
    dataset_path = "PlantDoc-Dataset"
    
    loader = PlantDataLoader(dataset_path)
    
    # Get dataset info
    info = loader.get_dataset_info()
    print("\n=== Dataset Information ===")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Total training images: {info.get('total_train_images', 'N/A')}")
    print(f"Total test images: {info.get('total_test_images', 'N/A')}")
    print("\nClass names:")
    for i, class_name in enumerate(info['class_names'], 1):
        train_count = info.get('train_counts', {}).get(class_name, 0)
        print(f"{i}. {class_name}: {train_count} images")
    
    # Create data generators
    print("\n=== Creating Data Generators ===")
    train_gen, val_gen, test_gen, class_names = loader.create_data_generators()
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    if test_gen:
        print(f"Test samples: {test_gen.samples}")
