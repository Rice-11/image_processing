#!/usr/bin/env python3
"""
Quick training script for Sign Language MNIST
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

def load_data():
    """Load and preprocess the training data"""
    print("Loading data...")
    
    # Load training data
    train_df = pd.read_csv('data/sign_mnist_train.csv')
    
    # Separate features and labels
    y = train_df.iloc[:, 0].values  # First column is label
    X = train_df.iloc[:, 1:].values  # Remaining columns are pixels
    
    # Reshape to 28x28 images
    X = X.reshape(-1, 28, 28, 1)
    
    # Normalize pixel values to 0-1
    X = X.astype('float32') / 255.0
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Classes: {len(np.unique(y))}")
    
    return X_train, X_val, y_train, y_val

def build_model():
    """Build CNN model"""
    print("Building model...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(24, activation='softmax')  # 24 classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model"""
    print("Training model...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_model(model):
    """Save the trained model"""
    os.makedirs('trained_models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'trained_models/quick_model_{timestamp}.h5'
    
    model.save(model_path)
    
    # Update latest model link
    latest_path = 'trained_models/latest_model.h5'
    if os.path.exists(latest_path):
        os.remove(latest_path)
    
    # Copy to latest
    import shutil
    shutil.copy2(model_path, latest_path)
    
    # Create class mapping
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
               'V', 'W', 'X', 'Y']
    
    mapping = {
        "classes": classes,
        "num_classes": 24,
        "class_to_index": {name: idx for idx, name in enumerate(classes)},
        "index_to_class": {str(idx): name for idx, name in enumerate(classes)}
    }
    
    import json
    with open('trained_models/latest_class_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Model saved: {model_path}")
    print("Updated latest_model.h5 and latest_class_mapping.json")
    
    return model_path

def main():
    print("QUICK TRAINING: Sign Language MNIST")
    print("=" * 50)
    
    # Load data
    X_train, X_val, y_train, y_val = load_data()
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    model_path = save_model(model)
    
    print("=" * 50)
    print("TRAINING COMPLETE!")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    if val_acc > 0.7:
        print("SUCCESS: Model trained successfully!")
        print("Run: python smart_asl_recognition.py")
    else:
        print("WARNING: Low accuracy. May need more training.")

if __name__ == "__main__":
    main()