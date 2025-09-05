#!/usr/bin/env python3
"""
Automatic training script for Sign Language MNIST
No user input required - runs with optimal settings
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_sign_mnist_model import SignLanguageMNISTTrainer

def main():
    print("AUTO TRAINING: Sign Language MNIST Model")
    print("=" * 50)
    
    # Initialize trainer
    trainer = SignLanguageMNISTTrainer(
        data_path="data/sign_mnist_train.csv",
        models_dir="trained_models"
    )
    
    # Load and prepare data
    print("Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_and_prepare_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build model
    print("Building model...")
    model = trainer.build_model()
    
    # Train model with optimal settings
    print("Starting training...")
    print("Using 30 epochs with data augmentation")
    
    history = trainer.train_model(
        model, 
        X_train, y_train, 
        X_val, y_val,
        epochs=30,
        use_augmentation=True
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_accuracy, test_loss = trainer.evaluate_model(model, X_test, y_test)
    
    # Save model and results
    print("Saving model...")
    timestamp = trainer.save_model_and_results(
        model, history, test_accuracy, test_loss
    )
    
    print("=" * 50)
    print("TRAINING COMPLETE!")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    if test_accuracy > 0.8:
        print("SUCCESS: Model trained successfully!")
        print("Run: python smart_asl_recognition.py")
    else:
        print("WARNING: Low accuracy. Consider more training data or epochs.")

if __name__ == "__main__":
    main()