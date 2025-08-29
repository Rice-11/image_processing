#!/usr/bin/env python3
"""
Sign Language Interpreter - Main Application
Real-time ASL letter recognition using deep learning and computer vision.
"""

import os
import sys
import argparse
import glob
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import SignLanguageTrainer
from src.camera_predictor import SignLanguagePredictor
from src.model import print_model_options, MODEL_CONFIGS

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🤟 SIGN LANGUAGE INTERPRETER")
    print("Real-time ASL Letter Recognition")
    print("=" * 60)
    print()

def interactive_mode():
    """Interactive mode for user-friendly experience"""
    print_banner()
    
    while True:
        print("What would you like to do?")
        print("1. 🎯 Quick Demo (Train lightweight model + Run camera)")
        print("2. 🏋️  Train a new model")
        print("3. 📸 Run real-time camera recognition")
        print("4. 📊 View model information")
        print("5. 🔧 List available models")
        print("6. ❌ Exit")
        print()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                run_quick_demo()
            elif choice == '2':
                interactive_training()
            elif choice == '3':
                interactive_prediction()
            elif choice == '4':
                print_model_options()
            elif choice == '5':
                list_available_models()
            elif choice == '6':
                print("👋 Thank you for using Sign Language Interpreter!")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Thank you for using Sign Language Interpreter!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

def interactive_training():
    """Interactive training workflow"""
    print("🏋️ MODEL TRAINING")
    print("-" * 30)
    
    # Choose model type
    print("Available model types:")
    for i, (model_type, config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"{i}. {model_type.upper()}: {config['description']}")
        print(f"   Expected accuracy: {config['accuracy']}")
        print(f"   Training time: {config['training_time']}")
        print()
    
    while True:
        try:
            choice = input("Choose model type (1-3): ").strip()
            model_types = list(MODEL_CONFIGS.keys())
            
            if choice in ['1', '2', '3']:
                model_type = model_types[int(choice) - 1]
                break
            else:
                print("Please enter 1, 2, or 3")
        except (ValueError, IndexError):
            print("Please enter 1, 2, or 3")
    
    # Choose epochs
    while True:
        try:
            epochs = input(f"Enter number of epochs (recommended for {model_type}: 50): ").strip()
            epochs = int(epochs) if epochs else 50
            if epochs > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Data augmentation
    augmentation = input("Use data augmentation? (Y/n): ").strip().lower()
    use_augmentation = augmentation != 'n'
    
    print(f"\n🚀 Starting training with:")
    print(f"   Model type: {model_type}")
    print(f"   Epochs: {epochs}")
    print(f"   Data augmentation: {'Yes' if use_augmentation else 'No'}")
    print()
    
    # Train model
    trainer = SignLanguageTrainer(model_type=model_type)
    trainer.run_full_training_pipeline(epochs=epochs, use_augmentation=use_augmentation)
    
    # Ask if user wants to test the model
    test_model = input("\nWould you like to test the trained model with camera? (Y/n): ").strip().lower()
    if test_model != 'n':
        # Find the most recent model
        model_files = glob.glob("models/*.h5")
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            predictor = SignLanguagePredictor(latest_model)
            predictor.run_real_time_prediction()

def interactive_prediction():
    """Interactive prediction workflow"""
    print("📸 REAL-TIME PREDICTION")
    print("-" * 30)
    
    # List available models
    model_files = glob.glob("models/*.h5")
    
    if not model_files:
        print("No trained models found!")
        print("Please train a model first using option 2 or run quick demo (option 1).")
        return
    
    print("Available models:")
    for i, model_file in enumerate(model_files, 1):
        model_name = os.path.basename(model_file)
        model_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        modified_time = datetime.fromtimestamp(os.path.getctime(model_file))
        print(f"{i}. {model_name}")
        print(f"   Size: {model_size:.1f} MB")
        print(f"   Created: {modified_time.strftime('%Y-%m-%d %H:%M')}")
        print()
    
    # Choose model
    while True:
        try:
            choice = input(f"Choose model (1-{len(model_files)}) or press Enter for latest: ").strip()
            
            if not choice:  # Use latest model
                model_path = max(model_files, key=os.path.getctime)
                break
            else:
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(model_files):
                    model_path = model_files[model_idx]
                    break
                else:
                    print(f"Please enter 1-{len(model_files)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Set confidence threshold
    while True:
        try:
            confidence = input("Confidence threshold (0.5-0.9, default 0.7): ").strip()
            confidence = float(confidence) if confidence else 0.7
            if 0.1 <= confidence <= 1.0:
                break
            else:
                print("Please enter a value between 0.1 and 1.0")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\n🚀 Starting camera with:")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Confidence threshold: {confidence}")
    print()
    
    # Run prediction
    predictor = SignLanguagePredictor(model_path, confidence)
    predictor.run_real_time_prediction()

def run_quick_demo():
    """Run quick demo with lightweight model"""
    print("🎯 QUICK DEMO")
    print("-" * 30)
    print("This will:")
    print("1. Download the Sign Language MNIST dataset")
    print("2. Train a lightweight model (20 epochs, ~5-10 minutes)")
    print("3. Start real-time camera recognition")
    print()
    
    confirm = input("Continue with quick demo? (Y/n): ").strip().lower()
    if confirm == 'n':
        return
    
    print("🚀 Starting quick demo...")
    
    # Train lightweight model with fewer epochs for speed
    trainer = SignLanguageTrainer(model_type='lightweight')
    trainer.run_full_training_pipeline(epochs=20, use_augmentation=True)
    
    # Find the trained model
    model_files = glob.glob("models/*.h5")
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        print(f"\n📸 Starting camera with trained model...")
        print("Place your hand in the green rectangle and make ASL letters!")
        
        predictor = SignLanguagePredictor(latest_model)
        predictor.run_real_time_prediction()
    else:
        print("Error: No model was created during training.")

def list_available_models():
    """List all available trained models"""
    print("🔧 AVAILABLE MODELS")
    print("-" * 30)
    
    model_files = glob.glob("models/*.h5")
    
    if not model_files:
        print("No trained models found.")
        print("Train a model using option 2 or run quick demo (option 1).")
        return
    
    print(f"Found {len(model_files)} trained model(s):\n")
    
    for i, model_file in enumerate(model_files, 1):
        model_name = os.path.basename(model_file)
        model_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        modified_time = datetime.fromtimestamp(os.path.getctime(model_file))
        
        print(f"{i}. {model_name}")
        print(f"   Path: {model_file}")
        print(f"   Size: {model_size:.1f} MB")
        print(f"   Created: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='Sign Language Interpreter - Real-time ASL Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive           # Interactive mode (recommended)
  python main.py --demo                 # Quick demo with lightweight model
  python main.py --train --model_type basic --epochs 50
  python main.py --predict --model_path models/model.h5
  python main.py --predict             # Use latest trained model
        """
    )
    
    # Main mode arguments
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode (recommended for beginners)')
    parser.add_argument('--demo', action='store_true',
                       help='Run quick demo (train lightweight model + camera)')
    
    # Training arguments
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--model_type', type=str, default='basic',
                       choices=['basic', 'advanced', 'lightweight'],
                       help='Type of model to train (default: basic)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation during training')
    
    # Prediction arguments
    parser.add_argument('--predict', action='store_true',
                       help='Run real-time prediction')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model (default: use latest)')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Confidence threshold for predictions (default: 0.7)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    
    # Utility arguments
    parser.add_argument('--list_models', action='store_true',
                       help='List all available trained models')
    
    args = parser.parse_args()
    
    # Handle no arguments - default to interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
        return
    
    try:
        if args.interactive:
            interactive_mode()
        
        elif args.demo:
            print_banner()
            run_quick_demo()
        
        elif args.train:
            print_banner()
            print(f"🏋️ Training {args.model_type} model for {args.epochs} epochs...")
            trainer = SignLanguageTrainer(model_type=args.model_type)
            trainer.run_full_training_pipeline(
                epochs=args.epochs,
                use_augmentation=not args.no_augmentation
            )
        
        elif args.predict:
            print_banner()
            
            # Determine model path
            if args.model_path:
                model_path = args.model_path
            else:
                # Use latest model
                model_files = glob.glob("models/*.h5")
                if not model_files:
                    print("No trained models found!")
                    print("Please train a model first using --train or --demo")
                    return
                model_path = max(model_files, key=os.path.getctime)
                print(f"Using latest model: {os.path.basename(model_path)}")
            
            predictor = SignLanguagePredictor(model_path, args.confidence_threshold)
            predictor.run_real_time_prediction(args.camera)
        
        elif args.list_models:
            print_banner()
            list_available_models()
        
        else:
            print("Please specify an action. Use --help for usage information.")
            print("For beginners, try: python main.py --interactive")
    
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled. Thank you for using Sign Language Interpreter!")
    
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("If this problem persists, please check your installation and try again.")

if __name__ == "__main__":
    main()