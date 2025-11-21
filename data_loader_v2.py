"""
Alternative Data Loader using tf.keras.utils.image_dataset_from_directory
This version handles long Windows paths better (>260 characters)
"""

import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
from pathlib import Path
import numpy as np


class PlantDataLoaderV2:
    """Handles loading and preprocessing of plant disease dataset with long path support"""
    
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader
        
        Args:
            dataset_path: Path to PlantDoc-Dataset folder
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        # Convert to absolute path with extended-length prefix for Windows
        self.dataset_path = Path(dataset_path).resolve()
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
    
    def create_data_augmentation(self):
        """Create data augmentation layer"""
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.2, 0.2),
        ])
    
    def preprocess_dataset(self, ds, augment=False):
        """Preprocess dataset with optional augmentation"""
        data_augmentation = self.create_data_augmentation() if augment else None
        
        def preprocess(image, label):
            # Resize to target size
            image = tf.image.resize(image, self.img_size)
            # Apply MobileNetV2 preprocessing
            image = preprocess_input(image)
            return image, label
        
        def augment_and_preprocess(image, label):
            # Resize first
            image = tf.image.resize(image, self.img_size)
            # Apply augmentation
            image = data_augmentation(image, training=True)
            # Apply MobileNetV2 preprocessing
            image = preprocess_input(image)
            return image, label
        
        if augment:
            ds = ds.map(augment_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Prefetch for performance
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return ds
    
    def create_data_generators(self, validation_split=0.2, seed=123):
        """
        Create data generators for training, validation, and testing
        
        Args:
            validation_split: Fraction of training data to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            train_dataset, validation_dataset, test_dataset, class_names
        """
        # Get class names
        self.get_class_names()
        
        # Load training dataset with validation split
        print(f"\nLoading training data from: {self.train_path}")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(self.train_path),
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode='categorical'
        )
        
        # Load validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(self.train_path),
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode='categorical'
        )
        
        # Get class names from the dataset
        class_names_from_ds = train_ds.class_names
        
        # Create class mapping
        class_indices = {name: idx for idx, name in enumerate(class_names_from_ds)}
        self.save_class_mapping(class_indices)
        
        # Apply preprocessing and augmentation
        print("Applying data augmentation to training set...")
        train_ds = self.preprocess_dataset(train_ds, augment=True)
        
        print("Applying preprocessing to validation set...")
        val_ds = self.preprocess_dataset(val_ds, augment=False)
        
        # Load test dataset if exists
        test_ds = None
        if self.test_path.exists():
            print(f"\nLoading test data from: {self.test_path}")
            test_ds = tf.keras.utils.image_dataset_from_directory(
                str(self.test_path),
                seed=seed,
                image_size=self.img_size,
                batch_size=self.batch_size,
                label_mode='categorical'
            )
            print("Applying preprocessing to test set...")
            test_ds = self.preprocess_dataset(test_ds, augment=False)
        
        # Calculate approximate sample counts
        train_samples = tf.data.experimental.cardinality(train_ds).numpy() * self.batch_size
        val_samples = tf.data.experimental.cardinality(val_ds).numpy() * self.batch_size
        
        print(f"\nData Generators Created:")
        print(f"  - Training samples: ~{train_samples}")
        print(f"  - Validation samples: ~{val_samples}")
        if test_ds:
            test_samples = tf.data.experimental.cardinality(test_ds).numpy() * self.batch_size
            print(f"  - Test samples: ~{test_samples}")
        
        return train_ds, val_ds, test_ds, class_names_from_ds
    
    def save_class_mapping(self, class_indices):
        """Save class name to index mapping"""
        # Reverse the mapping (index -> class_name)
        index_to_class = {v: k for k, v in class_indices.items()}
        
        mapping = {
            'class_indices': class_indices,
            'index_to_class': {str(k): v for k, v in index_to_class.items()},  # Convert keys to string for JSON
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
    
    loader = PlantDataLoaderV2(dataset_path)
    
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
    train_ds, val_ds, test_ds, class_names = loader.create_data_generators()
    print(f"\n✓ Datasets created successfully!")
