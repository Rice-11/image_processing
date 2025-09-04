#!/usr/bin/env python3
"""
ASL Model Training Script
Train a neural network to recognize ASL letters from hand landmarks
"""

import numpy as np
import json
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

class ASLModelTrainer:
    def __init__(self, data_dir="asl_training_data", models_dir="asl_models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        
        print("ASL Model Trainer Initialized")
        print(f"Data directory: {data_dir}")
        print(f"Models directory: {models_dir}")
    
    def load_training_data(self):
        """Load and combine all training data files"""
        print("\nLoading training data...")
        
        # Find all JSON data files
        data_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        if not data_files:
            raise FileNotFoundError(f"No training data found in {self.data_dir}")
        
        all_landmarks = []
        all_labels = []
        letter_counts = {}
        
        for file_path in data_files:
            print(f"Loading: {os.path.basename(file_path)}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract landmark data for each letter
            for letter, letter_data in data.items():
                if letter == 'metadata':
                    continue
                
                if 'landmarks' in letter_data:
                    landmarks = np.array(letter_data['landmarks'])
                    labels = [letter] * len(landmarks)
                    
                    all_landmarks.extend(landmarks)
                    all_labels.extend(labels)
                    
                    if letter in letter_counts:
                        letter_counts[letter] += len(landmarks)
                    else:
                        letter_counts[letter] = len(landmarks)
        
        if not all_landmarks:
            raise ValueError("No landmark data found in training files")
        
        # Convert to numpy arrays
        X = np.array(all_landmarks, dtype=np.float32)
        y = np.array(all_labels)
        
        # Shuffle data
        X, y = shuffle(X, y, random_state=42)
        
        print(f"\nData loaded successfully!")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]} (21 landmarks × 3 coordinates)")
        print(f"Unique letters: {len(np.unique(y))}")
        
        print("\nSamples per letter:")
        for letter in sorted(letter_counts.keys()):
            print(f"  {letter}: {letter_counts[letter]} samples")
        
        return X, y, letter_counts
    
    def preprocess_data(self, X, y, test_size=0.2, val_size=0.2):
        """Split and preprocess the data"""
        print(f"\nPreprocessing data...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, 
            stratify=y_categorical
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42,
            stratify=y_temp
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        # Normalize features (optional but often helpful)
        mean = np.mean(self.X_train, axis=0)
        std = np.std(self.X_train, axis=0)
        std[std == 0] = 1  # Prevent division by zero
        
        self.X_train = (self.X_train - mean) / std
        self.X_val = (self.X_val - mean) / std
        self.X_test = (self.X_test - mean) / std
        
        # Save normalization parameters
        self.normalization_params = {'mean': mean, 'std': std}
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def create_model(self, input_dim, num_classes):
        """Create the neural network model"""
        print(f"\nCreating model...")
        print(f"Input dimension: {input_dim}")
        print(f"Number of classes: {num_classes}")
        
        model = models.Sequential([
            # Input layer with normalization
            layers.Dense(256, input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def train_model(self, epochs=100, batch_size=32, patience=15):
        """Train the model"""
        print(f"\nStarting training...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Early stopping patience: {patience}")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print(f"\nEvaluating model...")
        
        # Predictions on test set
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            digits=4
        )
        
        print(f"\nDetailed Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.models_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {plot_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, evaluation_results):
        """Plot confusion matrix"""
        cm = evaluation_results['confusion_matrix']
        class_names = self.label_encoder.classes_
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('ASL Recognition Confusion Matrix')
        plt.xlabel('Predicted Letter')
        plt.ylabel('Actual Letter')
        
        # Save plot
        cm_path = os.path.join(self.models_dir, f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {cm_path}")
        
        plt.show()
    
    def save_model(self, evaluation_results):
        """Save the trained model and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.models_dir, f'asl_model_{timestamp}.h5')
        self.model.save(model_path)
        
        # Save label encoder
        encoder_path = os.path.join(self.models_dir, f'label_encoder_{timestamp}.json')
        encoder_data = {
            'classes': self.label_encoder.classes_.tolist(),
            'class_to_index': {cls: int(idx) for idx, cls in enumerate(self.label_encoder.classes_)}
        }
        
        with open(encoder_path, 'w') as f:
            json.dump(encoder_data, f, indent=2)
        
        # Save normalization parameters
        norm_path = os.path.join(self.models_dir, f'normalization_{timestamp}.json')
        norm_data = {
            'mean': self.normalization_params['mean'].tolist(),
            'std': self.normalization_params['std'].tolist()
        }
        
        with open(norm_path, 'w') as f:
            json.dump(norm_data, f, indent=2)
        
        # Save training metadata
        metadata_path = os.path.join(self.models_dir, f'training_metadata_{timestamp}.json')
        metadata = {
            'timestamp': timestamp,
            'model_path': model_path,
            'encoder_path': encoder_path,
            'normalization_path': norm_path,
            'test_accuracy': float(evaluation_results['accuracy']),
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'training_samples': len(self.X_train),
            'validation_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'feature_dimension': self.X_train.shape[1]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n" + "="*60)
        print("MODEL SAVED SUCCESSFULLY!")
        print("="*60)
        print(f"Model file: {model_path}")
        print(f"Label encoder: {encoder_path}")
        print(f"Normalization params: {norm_path}")
        print(f"Metadata: {metadata_path}")
        print(f"Test accuracy: {evaluation_results['accuracy']:.4f}")
        print("="*60)
        
        return {
            'model_path': model_path,
            'encoder_path': encoder_path,
            'normalization_path': norm_path,
            'metadata_path': metadata_path
        }
    
    def run_training_pipeline(self, epochs=100, batch_size=32):
        """Run the complete training pipeline"""
        print("="*60)
        print("ASL MODEL TRAINING PIPELINE")
        print("="*60)
        
        try:
            # 1. Load data
            X, y, letter_counts = self.load_training_data()
            
            # Check if we have enough data
            min_samples = min(letter_counts.values()) if letter_counts else 0
            if min_samples < 10:
                print(f"\nWARNING: Some letters have very few samples (minimum: {min_samples})")
                print("For best results, collect at least 30-50 samples per letter")
            
            # 2. Preprocess data
            self.preprocess_data(X, y)
            
            # 3. Create model
            self.model = self.create_model(
                input_dim=self.X_train.shape[1],
                num_classes=len(self.label_encoder.classes_)
            )
            
            # 4. Train model
            self.train_model(epochs=epochs, batch_size=batch_size)
            
            # 5. Evaluate model
            evaluation_results = self.evaluate_model()
            
            # 6. Visualize results
            self.plot_training_history()
            self.plot_confusion_matrix(evaluation_results)
            
            # 7. Save model
            saved_files = self.save_model(evaluation_results)
            
            print(f"\n🎉 Training pipeline completed successfully!")
            print(f"Final accuracy: {evaluation_results['accuracy']:.4f}")
            
            return saved_files
            
        except Exception as e:
            print(f"Training pipeline failed: {str(e)}")
            raise e

def main():
    """Main training function"""
    print("ASL Model Training")
    print("This script will train a neural network to recognize ASL letters")
    print()
    
    # Check for training data
    data_dir = "asl_training_data"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"❌ No training data found in '{data_dir}'")
        print("Please run 'asl_data_collector.py' first to collect training data")
        return
    
    # Initialize trainer
    trainer = ASLModelTrainer()
    
    # Get training parameters
    print("Training Configuration:")
    epochs = int(input("Enter number of epochs (default 100): ") or "100")
    batch_size = int(input("Enter batch size (default 32): ") or "32")
    
    print(f"\nStarting training with {epochs} epochs and batch size {batch_size}")
    input("Press Enter to continue...")
    
    # Run training
    trainer.run_training_pipeline(epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    main()