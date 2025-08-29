import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from data_loader import SignLanguageDataLoader
from model import create_model

class SignLanguageTrainer:
    def __init__(self, model_type='basic', model_save_dir='models'):
        self.model_type = model_type
        self.model_save_dir = model_save_dir
        self.data_loader = SignLanguageDataLoader()
        self.model_wrapper = None
        self.history = None
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and preprocess the dataset"""
        print("Loading Sign Language MNIST dataset...")
        
        # Download dataset
        train_df, test_df = self.data_loader.download_dataset()
        if train_df is None or test_df is None:
            raise Exception("Failed to download dataset")
        
        # Preprocess data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.data_loader.preprocess_data(train_df, test_df)
        
        print("Data loading completed successfully!")
        return True
    
    def create_and_compile_model(self):
        """Create and compile the CNN model"""
        print(f"Creating {self.model_type} model...")
        
        self.model_wrapper = create_model(
            model_type=self.model_type,
            input_shape=(28, 28, 1),
            num_classes=24
        )
        
        return self.model_wrapper
    
    def train_model(self, epochs=50, batch_size=64, use_augmentation=True):
        """Train the model with the prepared data"""
        if self.model_wrapper is None:
            raise Exception("Model not created. Call create_and_compile_model() first.")
        
        print(f"Starting training with {epochs} epochs...")
        
        # Prepare data generators
        if use_augmentation:
            train_datagen = self.data_loader.create_data_augmentation()
            train_generator = train_datagen.flow(
                self.X_train, self.y_train, 
                batch_size=batch_size
            )
        else:
            train_generator = None
        
        # Get callbacks
        model_save_path = os.path.join(
            self.model_save_dir, 
            f'sign_language_{self.model_type}_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        )
        
        callbacks = self.model_wrapper.get_callbacks(model_save_path)
        
        # Train the model
        if use_augmentation and train_generator:
            self.history = self.model_wrapper.model.fit(
                train_generator,
                steps_per_epoch=len(self.X_train) // batch_size,
                epochs=epochs,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model_wrapper.model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model_wrapper is None:
            raise Exception("Model not trained yet.")
        
        print("Evaluating model performance...")
        
        # Get predictions
        y_pred_proba = self.model_wrapper.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Calculate metrics
        evaluation = self.model_wrapper.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        test_loss = evaluation[0]
        test_accuracy = evaluation[1]
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            y_true, y_pred, 
            target_names=self.data_loader.class_names,
            digits=4
        ))
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'y_pred': y_pred,
            'y_true': y_true,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_training_history(self, save_plot=True):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot validation accuracy vs training accuracy difference
        if 'accuracy' in self.history.history and 'val_accuracy' in self.history.history:
            train_acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']
            acc_diff = [abs(t - v) for t, v in zip(train_acc, val_acc)]
            axes[1, 0].plot(acc_diff, label='Accuracy Gap (Train-Val)', color='orange')
            axes[1, 0].set_title('Training-Validation Accuracy Gap')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy Difference')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot learning rate (if available)
        if hasattr(self.model_wrapper.model.optimizer, 'lr'):
            axes[1, 1].plot(self.history.history.get('lr', []), label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.model_save_dir, f'training_history_{self.model_type}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {plot_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, evaluation_results, save_plot=True):
        """Plot confusion matrix"""
        y_true = evaluation_results['y_true']
        y_pred = evaluation_results['y_pred']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.data_loader.class_names,
            yticklabels=self.data_loader.class_names
        )
        plt.title(f'Confusion Matrix - {self.model_type.title()} Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_plot:
            cm_path = os.path.join(self.model_save_dir, f'confusion_matrix_{self.model_type}.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {cm_path}")
        
        plt.show()
    
    def save_training_results(self, evaluation_results):
        """Save training results and metadata"""
        results = {
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'test_accuracy': float(evaluation_results['test_accuracy']),
            'test_loss': float(evaluation_results['test_loss']),
            'training_history': {
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
                'loss': [float(x) for x in self.history.history['loss']],
                'val_loss': [float(x) for x in self.history.history['val_loss']]
            },
            'model_parameters': self.model_wrapper.model.count_params(),
            'class_names': self.data_loader.class_names
        }
        
        results_path = os.path.join(
            self.model_save_dir, 
            f'training_results_{self.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training results saved to: {results_path}")
        return results_path
    
    def visualize_sample_predictions(self, evaluation_results, num_samples=12):
        """Visualize sample predictions"""
        y_pred_proba = evaluation_results['y_pred_proba']
        y_true = evaluation_results['y_true']
        y_pred = evaluation_results['y_pred']
        
        # Get random samples
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Display image
            axes[i].imshow(self.X_test[idx].reshape(28, 28), cmap='gray')
            
            # Get predictions
            true_label = self.data_loader.class_names[y_true[idx]]
            pred_label = self.data_loader.class_names[y_pred[idx]]
            confidence = y_pred_proba[idx][y_pred[idx]]
            
            # Set title with color coding
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            axes[i].set_title(
                f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                color=color,
                fontsize=10
            )
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def run_full_training_pipeline(self, epochs=50, batch_size=64, use_augmentation=True):
        """Run the complete training pipeline"""
        print("Starting Sign Language Recognition Training Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Create model
            self.create_and_compile_model()
            
            # Step 3: Train model
            self.train_model(epochs=epochs, batch_size=batch_size, use_augmentation=use_augmentation)
            
            # Step 4: Evaluate model
            evaluation_results = self.evaluate_model()
            
            # Step 5: Visualize results
            self.plot_training_history()
            self.plot_confusion_matrix(evaluation_results)
            self.visualize_sample_predictions(evaluation_results)
            
            # Step 6: Save results
            results_path = self.save_training_results(evaluation_results)
            
            print("\nTraining pipeline completed successfully!")
            print(f"Final test accuracy: {evaluation_results['test_accuracy']:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            print(f"Training pipeline failed: {str(e)}")
            raise e

def main():
    """Main function for running training from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    parser.add_argument('--model_type', type=str, default='basic', 
                       choices=['basic', 'advanced', 'lightweight'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--no_augmentation', action='store_true', 
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SignLanguageTrainer(model_type=args.model_type)
    
    # Run training pipeline
    trainer.run_full_training_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation
    )

if __name__ == "__main__":
    main()