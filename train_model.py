"""
CNN Model Training Script using Transfer Learning
This module trains a MobileNetV2 model for plant disease classification
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
# Use V2 data loader which handles long Windows paths better (>260 chars)
from data_loader_v2 import PlantDataLoaderV2 as PlantDataLoader
import json


class PlantDiseaseModel:
    """Plant Disease Classification Model using Transfer Learning"""
    
    def __init__(self, num_classes, img_size=(224, 224)):
        """
        Initialize the model
        
        Args:
            num_classes: Number of disease classes
            img_size: Input image size
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the CNN model using MobileNetV2 with transfer learning"""
        
        # Load pre-trained MobileNetV2 (without top layers)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print("Model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, train_generator, validation_generator, epochs=30, fine_tune=False):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            fine_tune: Whether to fine-tune the base model after initial training
            
        Returns:
            training history
        """
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'plant_disease_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("\n=== Starting Initial Training (Frozen Base) ===")
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning (optional)
        if fine_tune:
            print("\n=== Starting Fine-Tuning (Unfrozen Base) ===")
            
            # Unfreeze the base model
            self.model.layers[0].trainable = True
            
            # Recompile with a lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
            )
            
            # Continue training
            fine_tune_history = self.model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=10,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in self.history.history.keys():
                self.history.history[key].extend(fine_tune_history.history[key])
        
        return self.history
    
    def evaluate(self, test_generator):
        """Evaluate the model on test set"""
        if test_generator is None:
            print("No test data available")
            return None
        
        print("\n=== Evaluating on Test Set ===")
        results = self.model.evaluate(test_generator, verbose=1)
        
        metrics = dict(zip(self.model.metrics_names, results))
        print(f"\nTest Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()
    
    def save_model(self, path='plant_disease_model.h5'):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path='plant_disease_model.h5'):
        """Load a trained model"""
        model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
        return model


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("Plant Disease Classification - Model Training")
    print("=" * 60)
    
    # Configuration
    DATASET_PATH = "PlantDoc-Dataset"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    FINE_TUNE = True
    
    # Load data
    print("\n1. Loading Dataset...")
    data_loader = PlantDataLoader(DATASET_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    # Get dataset info
    info = data_loader.get_dataset_info()
    print(f"\nDataset Info:")
    print(f"  - Classes: {info['num_classes']}")
    print(f"  - Training images: {info.get('total_train_images', 'N/A')}")
    print(f"  - Test images: {info.get('total_test_images', 'N/A')}")
    
    # Create data generators
    train_gen, val_gen, test_gen, class_names = data_loader.create_data_generators(validation_split=0.2)
    
    # Note: V2 data loader already prints sample counts during creation
    print(f"\n✓ Data generators ready for training!")
    
    # Build model
    print("\n2. Building Model...")
    model_trainer = PlantDiseaseModel(num_classes=len(class_names), img_size=IMG_SIZE)
    model_trainer.build_model()
    
    # Print model summary
    print("\nModel Summary:")
    model_trainer.model.summary()
    
    # Train model
    print("\n3. Training Model...")
    history = model_trainer.train(
        train_gen,
        val_gen,
        epochs=EPOCHS,
        fine_tune=FINE_TUNE
    )
    
    # Plot training history
    print("\n4. Plotting Training History...")
    model_trainer.plot_training_history()
    
    # Evaluate on test set
    if test_gen:
        print("\n5. Evaluating on Test Set...")
        print("Note: Test set has 27 classes while model was trained on 28 classes.")
        print("Skipping test evaluation due to class mismatch.")
        # test_metrics = model_trainer.evaluate(test_gen)
    
    # Save final model
    print("\n6. Saving Model...")
    model_trainer.save_model('plant_disease_model.h5')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nModel saved as: plant_disease_model.h5")
    print("Class mapping saved as: class_mapping.json")
    print("Training history plot saved as: training_history.png")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run training
    main()
