#!/usr/bin/env python3
"""
Quick Start Script for Sign Language Recognition
Get your gesture recognition system running in minutes!
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    print("=" * 60)
    print("🚀 QUICK START - SIGN LANGUAGE RECOGNITION")
    print("Get your system running in 3 steps!")
    print("=" * 60)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['cv2', 'mediapipe', 'tensorflow', 'numpy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mediapipe':
                import mediapipe
            elif package == 'tensorflow':
                import tensorflow
            elif package == 'numpy':
                import numpy
            elif package == 'sklearn':
                import sklearn
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing packages"""
    if not missing_packages:
        return True
    
    print("📦 Installing missing packages...")
    
    # Map package names to pip install names
    pip_names = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn'
    }
    
    for package in missing_packages:
        pip_name = pip_names.get(package, package)
        print(f"Installing {pip_name}...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', pip_name], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install {pip_name}")
            return False
        else:
            print(f"✅ {pip_name} installed successfully")
    
    return True

def create_sample_data():
    """Create sample gesture data for quick testing"""
    from src.mediapipe_hand_tracker import MediaPipeHandTracker
    
    print("📸 Creating sample gesture data...")
    print("This will collect a few samples of basic gestures for testing.")
    
    # Create directories
    os.makedirs("gesture_data/static", exist_ok=True)
    os.makedirs("gesture_data/dynamic", exist_ok=True)
    
    # Create some basic sample data
    sample_static_data = {
        "A": {
            "label": "A",
            "type": "static",
            "samples": [
                [0.1, 0.2, 0.0] * 21,  # Simplified landmark data
                [0.1, 0.2, 0.0] * 21,
                [0.1, 0.2, 0.0] * 21,
                [0.1, 0.2, 0.0] * 21,
                [0.1, 0.2, 0.0] * 21,
            ],
            "num_samples": 5,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        },
        "B": {
            "label": "B", 
            "type": "static",
            "samples": [
                [0.2, 0.3, 0.0] * 21,
                [0.2, 0.3, 0.0] * 21,
                [0.2, 0.3, 0.0] * 21,
                [0.2, 0.3, 0.0] * 21,
                [0.2, 0.3, 0.0] * 21,
            ],
            "num_samples": 5,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    }
    
    sample_dynamic_data = {
        "wave": {
            "label": "wave",
            "type": "dynamic",
            "sequence": [[0.1 + i*0.01, 0.2, 0.0] * 21 for i in range(30)],
            "sequence_length": 30,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        },
        "hello": {
            "label": "hello",
            "type": "dynamic", 
            "sequence": [[0.2 + i*0.005, 0.3, 0.0] * 21 for i in range(25)],
            "sequence_length": 25,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    }
    
    # Save static data
    for gesture, data in sample_static_data.items():
        filename = f"gesture_data/static/{gesture}_sample.json"
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    # Save dynamic data  
    for gesture, data in sample_dynamic_data.items():
        filename = f"gesture_data/dynamic/{gesture}_sample.json"
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    print("✅ Sample gesture data created!")
    print(f"Static gestures: {list(sample_static_data.keys())}")
    print(f"Dynamic gestures: {list(sample_dynamic_data.keys())}")
    
    return True

def train_models():
    """Train the gesture recognition models"""
    print("🏋️ Training gesture recognition models...")
    
    try:
        from src.gesture_trainer import GestureTrainer
        
        trainer = GestureTrainer()
        success = trainer.train_both_models(static_epochs=10, dynamic_epochs=20)  # Reduced epochs for speed
        
        if success:
            print("✅ Models trained successfully!")
            return True
        else:
            print("❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def run_prediction():
    """Run real-time gesture prediction"""
    print("🎯 Starting real-time gesture recognition...")
    
    try:
        from src.unified_predictor import UnifiedGesturePredictor
        
        predictor = UnifiedGesturePredictor()
        predictor.run_real_time_prediction()
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

def main():
    print_banner()
    
    print("STEP 1: Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"Missing packages: {missing}")
        if input("Install missing packages? (y/n): ").lower() == 'y':
            if not install_dependencies(missing):
                print("❌ Dependency installation failed. Please install manually.")
                return
        else:
            print("❌ Cannot proceed without required packages.")
            return
    else:
        print("✅ All dependencies are installed!")
    
    print("\nSTEP 2: Setting up sample data...")
    choice = input("Create sample data for quick testing? (y/n): ").lower()
    if choice == 'y':
        create_sample_data()
    
    print("\nSTEP 3: Training models...")
    choice = input("Train gesture recognition models? (y/n): ").lower() 
    if choice == 'y':
        if not train_models():
            print("❌ Cannot proceed without trained models.")
            return
    
    print("\n🎉 Setup complete!")
    print("\nWhat would you like to do next?")
    print("1. 📸 Collect your own gesture data")
    print("2. 🎯 Run real-time gesture recognition")  
    print("3. 🏋️ Train models with more data")
    print("4. ❌ Exit")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            print("Starting data collection...")
            from src.mediapipe_hand_tracker import MediaPipeHandTracker
            tracker = MediaPipeHandTracker()
            tracker.run_data_collection()
            break
            
        elif choice == '2':
            run_prediction()
            break
            
        elif choice == '3':
            train_models()
            break
            
        elif choice == '4':
            print("👋 Thank you for using Sign Language Recognition!")
            break
            
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()