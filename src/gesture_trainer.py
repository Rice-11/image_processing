import numpy as np
import json
import os
import glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class GestureTrainer:
    def __init__(self):
        self.static_model = None
        self.dynamic_model = None
        self.static_encoder = LabelEncoder()
        self.dynamic_encoder = LabelEncoder()
        self.models_dir = "gesture_models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_static_data(self, data_dir="gesture_data/static"):
        """Load static gesture data"""
        print("Loading static gesture data...")
        
        X, y = [], []
        
        for json_file in glob.glob(f"{data_dir}/*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            label = data['label']
            samples = data['samples']
            
            for sample in samples:
                X.append(sample)
                y.append(label)
        
        if not X:
            print("No static data found! Please collect some data first.")
            return None, None
            
        X = np.array(X)
        y = np.array(y)
        
        print(f"Loaded {len(X)} static samples with {len(np.unique(y))} different gestures")
        print(f"Gestures: {np.unique(y)}")
        
        return X, y
    
    def load_dynamic_data(self, data_dir="gesture_data/dynamic"):
        """Load dynamic gesture data"""
        print("Loading dynamic gesture data...")
        
        X, y = [], []
        max_length = 0
        
        # First pass: find max sequence length
        for json_file in glob.glob(f"{data_dir}/*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            max_length = max(max_length, data['sequence_length'])
        
        # Second pass: load and pad sequences
        for json_file in glob.glob(f"{data_dir}/*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            label = data['label']
            sequence = data['sequence']
            
            # Pad sequence to max_length
            padded_sequence = sequence + [[0]*63] * (max_length - len(sequence))
            
            X.append(padded_sequence)
            y.append(label)
        
        if not X:
            print("No dynamic data found! Please collect some data first.")
            return None, None, 0
            
        X = np.array(X)
        y = np.array(y)
        
        print(f"Loaded {len(X)} dynamic sequences with {len(np.unique(y))} different gestures")
        print(f"Gestures: {np.unique(y)}")
        print(f"Max sequence length: {max_length}")
        
        return X, y, max_length
    
    def create_static_model(self, input_dim, num_classes):
        """Create neural network for static gesture recognition"""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_dynamic_model(self, sequence_length, feature_dim, num_classes):
        """Create LSTM model for dynamic gesture recognition"""
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, feature_dim)),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_static_model(self, epochs=30):
        """Train static gesture recognition model"""
        print("=== TRAINING STATIC GESTURE MODEL ===")
        
        X, y = self.load_static_data()
        if X is None:
            return False
            
        # Encode labels
        y_encoded = self.static_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
        )
        
        # Create and train model
        self.static_model = self.create_static_model(X.shape[1], len(self.static_encoder.classes_))
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        history = self.static_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model and encoder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.models_dir}/static_model_{timestamp}.h5"
        encoder_path = f"{self.models_dir}/static_encoder_{timestamp}.npy"
        
        self.static_model.save(model_path)
        np.save(encoder_path, self.static_encoder.classes_)
        
        print(f"Static model saved to {model_path}")
        print(f"Final accuracy: {history.history['val_accuracy'][-1]:.3f}")
        
        return True
    
    def train_dynamic_model(self, epochs=50):
        """Train dynamic gesture recognition model"""
        print("=== TRAINING DYNAMIC GESTURE MODEL ===")
        
        X, y, max_length = self.load_dynamic_data()
        if X is None:
            return False
            
        # Encode labels
        y_encoded = self.dynamic_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
        )
        
        # Create and train model
        self.dynamic_model = self.create_dynamic_model(
            max_length, 63, len(self.dynamic_encoder.classes_)
        )
        
        print(f"Training on {len(X_train)} sequences, testing on {len(X_test)} sequences")
        
        history = self.dynamic_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model and encoder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.models_dir}/dynamic_model_{timestamp}.h5"
        encoder_path = f"{self.models_dir}/dynamic_encoder_{timestamp}.npy"
        
        self.dynamic_model.save(model_path)
        np.save(encoder_path, self.dynamic_encoder.classes_)
        
        print(f"Dynamic model saved to {model_path}")
        print(f"Final accuracy: {history.history['val_accuracy'][-1]:.3f}")
        
        return True
    
    def train_both_models(self, static_epochs=30, dynamic_epochs=50):
        """Train both static and dynamic models"""
        print("=== TRAINING BOTH GESTURE RECOGNITION MODELS ===")
        
        success_static = self.train_static_model(static_epochs)
        success_dynamic = self.train_dynamic_model(dynamic_epochs)
        
        if success_static and success_dynamic:
            print("\n🎉 Both models trained successfully!")
            print("Ready for real-time gesture recognition!")
            return True
        elif success_static:
            print("\n✅ Static model trained successfully!")
            print("⚠️  Dynamic model training failed or no dynamic data available")
            return True
        elif success_dynamic:
            print("\n✅ Dynamic model trained successfully!")
            print("⚠️  Static model training failed or no static data available")
            return True
        else:
            print("\n❌ Both model training failed. Please collect gesture data first.")
            return False

def main():
    """Quick training interface"""
    print("=== GESTURE MODEL TRAINER ===")
    print("1. Train static gesture model only")
    print("2. Train dynamic gesture model only") 
    print("3. Train both models (recommended)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    trainer = GestureTrainer()
    
    if choice == '1':
        trainer.train_static_model()
    elif choice == '2':
        trainer.train_dynamic_model()
    elif choice == '3':
        trainer.train_both_models()
    else:
        print("Invalid choice. Training both models...")
        trainer.train_both_models()

if __name__ == "__main__":
    main()