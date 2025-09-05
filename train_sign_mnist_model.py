#!/usr/bin/env python3
"""
Professional Sign Language MNIST Model Training Script
Trains a CNN model on the Sign Language MNIST dataset for ASL letter recognition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SignLanguageMNISTTrainer:
    def __init__(self, data_path="data/sign_mnist_train.csv", models_dir="trained_models"):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # ASL alphabet mapping (J and Z are excluded from Sign Language MNIST)
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                           'V', 'W', 'X', 'Y']
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.history = None
        
        print("Sign Language MNIST Trainer Initialized")
        print(f"Data path: {data_path}")
        print(f"Models directory: {models_dir}")
        print(f"Target classes: {len(self.class_names)} letters")
    
    def load_and_preprocess_data(self, validation_split=0.2, test_split=0.1):
        """Load and preprocess the Sign Language MNIST dataset"""
        print("\nLoading Sign Language MNIST dataset...")
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        # Load the CSV file
        print(f"Reading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
        
        # Extract labels (first column) and pixel values (remaining columns)
        labels = df.iloc[:, 0].values
        pixels = df.iloc[:, 1:].values
        
        print(f"Labels shape: {labels.shape}")
        print(f"Pixels shape: {pixels.shape}")
        print(f"Unique labels: {sorted(np.unique(labels))}")
        print(f"Expected pixel features: 784 (28x28)")
        
        # Verify data integrity
        if pixels.shape[1] != 784:
            raise ValueError(f"Expected 784 pixel features, got {pixels.shape[1]}")
        
        # Reshape pixel data to 28x28x1 images
        print("\nReshaping images...")
        images = pixels.reshape(-1, 28, 28, 1).astype('float32')
        
        # Normalize pixel values to [0, 1]
        images = images / 255.0
        print(f"Images normalized to range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Convert labels to categorical
        print("\nProcessing labels...")
        labels_categorical = to_categorical(labels, num_classes=len(self.class_names))
        
        # Display class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\nClass distribution:")
        for label, count in zip(unique_labels, counts):
            label_idx = int(label)  # Convert to int for indexing
            class_name = self.class_names[label_idx] if label_idx < len(self.class_names) else f"Unknown({label_idx})"
            print(f"  {class_name} ({label_idx}): {count} samples")
        
        # Split data: first separate test set, then split remaining into train/val
        print(f"\nSplitting data (test: {test_split}, validation: {validation_split})...")
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            images, labels_categorical, 
            test_size=test_split, 
            random_state=42, 
            stratify=labels_categorical
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = validation_split / (1 - test_split)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"Training set: {self.X_train.shape} images, {self.y_train.shape} labels")
        print(f"Validation set: {self.X_val.shape} images, {self.y_val.shape} labels")
        print(f"Test set: {self.X_test.shape} images, {self.y_test.shape} labels")
        
        # Visualize sample data
        self.visualize_sample_data()
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def visualize_sample_data(self, num_samples=12):
        """Visualize sample images from the dataset"""
        print("\nVisualizing sample data...")
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.ravel()
        
        # Get random samples from training data
        indices = np.random.choice(len(self.X_train), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            # Display image
            axes[i].imshow(self.X_train[idx].squeeze(), cmap='gray')
            
            # Get label
            label_idx = np.argmax(self.y_train[idx])
            class_name = self.class_names[label_idx]
            
            axes[i].set_title(f'Class: {class_name}')
            axes[i].axis('off')
        
        plt.suptitle('Sample ASL Letters from Training Data', fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.models_dir, 'sample_data_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Sample visualization saved: {viz_path}")
        plt.show()
    
    def create_cnn_model(self):
        """Create an optimized CNN model for Sign Language recognition"""
        print("\nCreating CNN model...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense Classification Head
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile model with advanced optimizer
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("Model Architecture:")
        model.summary()
        
        # Calculate and display model parameters
        total_params = model.count_params()
        print(f"\nModel Parameters:")
        print(f"Total parameters: {total_params:,}")
        print(f"Estimated model size: {total_params * 4 / (1024**2):.2f} MB")
        
        return model
    
    def create_data_generators(self):
        """Create data generators with augmentation for training"""
        print("\nCreating data generators with augmentation...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            self.X_train, self.y_train,
            batch_size=64,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            self.X_val, self.y_val,
            batch_size=64,
            shuffle=False
        )
        
        print("Data generators created successfully")
        return train_generator, val_generator
    
    def train_model(self, epochs=100, use_augmentation=True):
        """Train the CNN model"""
        print(f"\nStarting model training...")
        print(f"Epochs: {epochs}")
        print(f"Data augmentation: {use_augmentation}")
        
        # Create model
        self.model = self.create_cnn_model()
        
        # Setup callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(self.models_dir, f'sign_mnist_model_{timestamp}.h5')
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        if use_augmentation:
            # Train with data augmentation
            train_generator, val_generator = self.create_data_generators()
            
            self.history = self.model.fit(
                train_generator,
                steps_per_epoch=len(self.X_train) // 64,
                validation_data=val_generator,
                validation_steps=len(self.X_val) // 64,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=1
            )
        else:
            # Train without augmentation
            self.history = self.model.fit(
                self.X_train, self.y_train,
                batch_size=64,
                epochs=epochs,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks_list,
                verbose=1
            )
        
        print("Training completed successfully!")
        return self.history
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\nEvaluating model performance...")
        
        # Evaluate on test set
        test_loss, test_accuracy, test_top_k = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Top-K Accuracy: {test_top_k:.4f} ({test_top_k*100:.2f}%)")
        
        # Get predictions for detailed analysis
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Classification report
        print("\nDetailed Classification Report:")
        report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top_k_accuracy': test_top_k,
            'predictions': y_pred,
            'true_labels': y_true_classes,
            'predicted_labels': y_pred_classes,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        print("\nPlotting training history...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot training & validation loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot top-k accuracy
        if 'top_k_categorical_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_k_categorical_accuracy'], 
                           label='Training Top-K Accuracy')
            axes[1, 0].plot(self.history.history['val_top_k_categorical_accuracy'], 
                           label='Validation Top-K Accuracy')
            axes[1, 0].set_title('Top-K Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-K Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot learning rate if available
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.models_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {plot_path}")
        plt.show()
    
    def plot_confusion_matrix(self, evaluation_results):
        """Plot confusion matrix"""
        cm = evaluation_results['confusion_matrix']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Sign Language MNIST - Confusion Matrix')
        plt.xlabel('Predicted Letter')
        plt.ylabel('True Letter')
        
        # Save plot
        cm_path = os.path.join(self.models_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {cm_path}")
        plt.show()
    
    def save_model_artifacts(self, evaluation_results):
        """Save all model artifacts for deployment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\nSaving model artifacts...")
        
        # Save the trained model
        model_path = os.path.join(self.models_dir, f'sign_mnist_model_{timestamp}.h5')
        self.model.save(model_path)
        print(f"Model saved: {model_path}")
        
        # Save class mapping
        class_mapping = {
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'class_to_index': {name: idx for idx, name in enumerate(self.class_names)},
            'index_to_class': {idx: name for idx, name in enumerate(self.class_names)}
        }
        
        mapping_path = os.path.join(self.models_dir, f'class_mapping_{timestamp}.json')
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        print(f"Class mapping saved: {mapping_path}")
        
        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'model_path': model_path,
            'class_mapping_path': mapping_path,
            'dataset_path': self.data_path,
            'model_architecture': 'CNN',
            'input_shape': [28, 28, 1],
            'num_classes': len(self.class_names),
            'classes': self.class_names,
            'training_samples': len(self.X_train),
            'validation_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'test_accuracy': float(evaluation_results['test_accuracy']),
            'test_loss': float(evaluation_results['test_loss']),
            'preprocessing': {
                'input_range': '[0, 1]',
                'normalization': 'divide by 255',
                'required_shape': '(28, 28, 1)',
                'color_mode': 'grayscale'
            }
        }
        
        metadata_path = os.path.join(self.models_dir, f'training_metadata_{timestamp}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Training metadata saved: {metadata_path}")
        
        # Create a "latest" symlink for easy access
        latest_model_path = os.path.join(self.models_dir, 'latest_model.h5')
        latest_mapping_path = os.path.join(self.models_dir, 'latest_class_mapping.json')
        latest_metadata_path = os.path.join(self.models_dir, 'latest_metadata.json')
        
        try:
            # Remove existing symlinks if they exist
            for path in [latest_model_path, latest_mapping_path, latest_metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            # Create new symlinks (or copies on Windows)
            import shutil
            shutil.copy2(model_path, latest_model_path)
            shutil.copy2(mapping_path, latest_mapping_path)
            shutil.copy2(metadata_path, latest_metadata_path)
            
            print("Latest model artifacts created for easy deployment access")
        except Exception as e:
            print(f"Note: Could not create 'latest' shortcuts: {e}")
        
        return {
            'model_path': model_path,
            'class_mapping_path': mapping_path,
            'metadata_path': metadata_path,
            'timestamp': timestamp
        }
    
    def run_complete_training_pipeline(self, epochs=100, use_augmentation=True):
        """Run the complete training pipeline"""
        print("="*70)
        print("SIGN LANGUAGE MNIST TRAINING PIPELINE")
        print("="*70)
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Train model
            self.train_model(epochs=epochs, use_augmentation=use_augmentation)
            
            # Step 3: Evaluate model
            evaluation_results = self.evaluate_model()
            
            # Step 4: Visualize results
            self.plot_training_history()
            self.plot_confusion_matrix(evaluation_results)
            
            # Step 5: Save all artifacts
            saved_artifacts = self.save_model_artifacts(evaluation_results)
            
            print("\n" + "="*70)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"Final Test Accuracy: {evaluation_results['test_accuracy']:.4f} ({evaluation_results['test_accuracy']*100:.2f}%)")
            print(f"Model saved: {saved_artifacts['model_path']}")
            print("Ready for deployment with live prediction script!")
            print("="*70)
            
            return saved_artifacts
            
        except Exception as e:
            print(f"\n❌ Training pipeline failed: {str(e)}")
            raise e

def main():
    """Main training function"""
    print("Sign Language MNIST Model Training")
    print("Training a CNN model for ASL letter recognition")
    print()
    
    # Check for dataset
    data_path = "data/sign_mnist_train.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset file not found: {data_path}")
        print("Please ensure the Sign Language MNIST dataset is in the correct location")
        return
    
    # Initialize trainer
    trainer = SignLanguageMNISTTrainer(data_path)
    
    # Get training parameters
    print("Training Configuration:")
    epochs = int(input("Enter number of epochs (default 50): ") or "50")
    use_aug = input("Use data augmentation? (Y/n): ").strip().lower() != 'n'
    
    print(f"\nStarting training with:")
    print(f"  Epochs: {epochs}")
    print(f"  Data augmentation: {use_aug}")
    print()
    
    input("Press Enter to start training...")
    
    # Run training pipeline
    trainer.run_complete_training_pipeline(epochs=epochs, use_augmentation=use_aug)

if __name__ == "__main__":
    main()