#!/usr/bin/env python3
"""
ASL Landmark Model Trainer
Trains ML models on normalized 3D hand landmark data
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using scikit-learn models only.")

class LandmarkModelTrainer:
    def __init__(self, data_file="asl_landmark_data.json", model_dir="landmark_models"):
        self.data_file = Path(data_file)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
        # Model containers
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0.0
        
        print("ASL Landmark Model Trainer Initialized")
        print("=" * 50)
    
    def load_data(self):
        """Load and prepare training data from JSON file"""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        print(f"Loading data from: {self.data_file}")
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        # Extract data and metadata
        if 'data' in data:
            landmark_data = data['data']
            metadata = data.get('metadata', {})
            print(f"Data collected on: {metadata.get('collection_date', 'Unknown')}")
            print(f"Feature dimension: {metadata.get('feature_dimension', 'Unknown')}")
        else:
            landmark_data = data  # Backward compatibility
        
        # Prepare training arrays
        features = []
        labels = []
        
        # Only process letter data (skip metadata and other non-letter keys)
        valid_letters = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                            'V', 'W', 'X', 'Y'])
        
        for key, samples in landmark_data.items():
            # Skip non-letter keys like 'metadata'
            if key not in valid_letters:
                print(f"Skipping non-letter key: {key}")
                continue
                
            if not isinstance(samples, list):
                print(f"Skipping invalid data for {key}: not a list")
                continue
                
            print(f"Loading {key}: {len(samples)} samples")
            
            for sample in samples:
                try:
                    if isinstance(sample, dict):
                        # New format with metadata
                        if 'features' in sample:
                            features.append(sample['features'])
                            labels.append(key)
                        else:
                            print(f"Warning: Sample missing 'features' key in {key}")
                    elif isinstance(sample, list):
                        # Old format (just features)
                        features.append(sample)
                        labels.append(key)
                    else:
                        print(f"Warning: Invalid sample format in {key}")
                except Exception as e:
                    print(f"Warning: Error processing sample in {key}: {e}")
        
        # Check if we have any data
        if not features:
            raise ValueError("No training data found! Please collect data first with: python landmark_collector.py")
        
        self.X = np.array(features, dtype=np.float32)
        self.y = np.array(labels)
        
        # Basic data validation
        if len(self.X) == 0:
            raise ValueError("No valid samples found!")
        
        if len(set(labels)) < 2:
            raise ValueError("Need at least 2 different letters to train. Current letters: " + str(set(labels)))
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"\\nDataset Summary:")
        print(f"Total samples: {len(self.X)}")
        print(f"Feature dimension: {self.X.shape[1]}")
        print(f"Classes: {len(self.label_encoder.classes_)}")
        print(f"Class distribution:")
        
        unique, counts = np.unique(self.y, return_counts=True)
        for letter, count in zip(unique, counts):
            print(f"  {letter}: {count} samples")
        
        # Check for minimum samples per class
        min_samples = min(counts)
        if min_samples < 5:
            print(f"\\nWARNING: Some letters have very few samples (minimum: {min_samples})")
            print("Consider collecting more data for better accuracy")
        
        # Validate data quality
        self.validate_data()
        
    def validate_data(self):
        """Validate data quality and consistency"""
        print("\\nValidating data quality...")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(self.X)) or np.any(np.isinf(self.X)):
            print("WARNING: Found NaN or infinite values in data")
            
            # Remove problematic samples
            valid_mask = ~(np.isnan(self.X).any(axis=1) | np.isinf(self.X).any(axis=1))
            self.X = self.X[valid_mask]
            self.y_encoded = self.y_encoded[valid_mask]
            self.y = self.y[valid_mask]
            print(f"Removed invalid samples. New size: {len(self.X)}")
        
        # Check feature variance
        feature_var = np.var(self.X, axis=0)
        low_variance_features = np.sum(feature_var < 1e-6)
        
        if low_variance_features > 0:
            print(f"WARNING: {low_variance_features} features have very low variance")
        
        # Data distribution analysis
        print(f"Feature range: [{np.min(self.X):.3f}, {np.max(self.X):.3f}]")
        print(f"Feature mean: {np.mean(self.X):.3f}")
        print(f"Feature std: {np.std(self.X):.3f}")
        print("✓ Data validation complete")
    
    def train_random_forest(self):
        """Train Random Forest classifier with hyperparameter tuning"""
        print("\\nTraining Random Forest Classifier...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        print("Performing hyperparameter optimization...")
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(self.X, self.y_encoded)
        
        best_rf = grid_search.best_estimator_
        best_score = grid_search.best_score_
        
        print(f"Best RF parameters: {grid_search.best_params_}")
        print(f"Best RF cross-validation score: {best_score:.4f}")
        
        self.models['random_forest'] = {
            'model': best_rf,
            'cv_score': best_score,
            'params': grid_search.best_params_
        }
        
        return best_rf, best_score
    
    def train_svm(self):
        """Train Support Vector Machine"""
        print("\\nTraining SVM Classifier...")
        
        # SVM hyperparameters
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly']
        }
        
        svm = SVC(random_state=42, probability=True)
        
        print("Performing SVM hyperparameter optimization...")
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(self.X, self.y_encoded)
        
        best_svm = grid_search.best_estimator_
        best_score = grid_search.best_score_
        
        print(f"Best SVM parameters: {grid_search.best_params_}")
        print(f"Best SVM cross-validation score: {best_score:.4f}")
        
        self.models['svm'] = {
            'model': best_svm,
            'cv_score': best_score,
            'params': grid_search.best_params_
        }
        
        return best_svm, best_score
    
    def train_neural_network_sklearn(self):
        """Train Neural Network using scikit-learn"""
        print("\\nTraining Neural Network (scikit-learn)...")
        
        # MLP hyperparameters
        param_grid = {
            'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100), (300, 150, 75)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        mlp = MLPClassifier(random_state=42, max_iter=1000)
        
        print("Performing MLP hyperparameter optimization...")
        grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(self.X, self.y_encoded)
        
        best_mlp = grid_search.best_estimator_
        best_score = grid_search.best_score_
        
        print(f"Best MLP parameters: {grid_search.best_params_}")
        print(f"Best MLP cross-validation score: {best_score:.4f}")
        
        self.models['neural_network_sklearn'] = {
            'model': best_mlp,
            'cv_score': best_score,
            'params': grid_search.best_params_
        }
        
        return best_mlp, best_score
    
    def train_deep_neural_network(self):
        """Train Deep Neural Network using TensorFlow/Keras"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping deep neural network.")
            return None, 0.0
        
        print("\\nTraining Deep Neural Network (TensorFlow)...")
        
        # Prepare data for Keras
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        # Convert to categorical
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
        
        # Build model
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.X.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        print("Training deep neural network...")
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
        print(f"Deep NN validation accuracy: {val_accuracy:.4f}")
        
        self.models['deep_neural_network'] = {
            'model': model,
            'cv_score': val_accuracy,
            'history': history.history
        }
        
        return model, val_accuracy
    
    def evaluate_models(self):
        """Evaluate all trained models and select the best one"""
        print("\\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        for model_name, model_info in self.models.items():
            if model_name == 'deep_neural_network' and TENSORFLOW_AVAILABLE:
                # Handle Keras model differently
                y_pred_proba = model_info['model'].predict(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                # Scikit-learn models
                model = model_info['model']
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\\n{model_name.upper().replace('_', ' ')}:")
            print(f"Cross-validation score: {model_info['cv_score']:.4f}")
            print(f"Test accuracy: {accuracy:.4f}")
            
            # Update best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model_info['model']
                self.best_model_name = model_name
            
            # Detailed classification report
            y_test_labels = self.label_encoder.inverse_transform(y_test)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            
            print("\\nClassification Report:")
            print(classification_report(y_test_labels, y_pred_labels))
        
        print(f"\\n🏆 BEST MODEL: {self.best_model_name.replace('_', ' ').upper()}")
        print(f"🎯 BEST ACCURACY: {self.best_accuracy:.4f}")
        
        return self.best_model, self.best_model_name, self.best_accuracy
    
    def save_best_model(self):
        """Save the best model and label encoder"""
        if self.best_model is None:
            print("No model to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        if self.best_model_name == 'deep_neural_network':
            model_path = self.model_dir / f"asl_landmark_model_keras_{timestamp}.h5"
            self.best_model.save(model_path)
            
            # Also save as latest
            latest_model_path = self.model_dir / "asl_landmark_model_keras.h5"
            self.best_model.save(latest_model_path)
        else:
            model_path = self.model_dir / f"asl_landmark_model_sklearn_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            # Also save as latest
            latest_model_path = self.model_dir / "asl_landmark_model_sklearn.pkl"
            with open(latest_model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
        
        # Save label encoder
        encoder_path = self.model_dir / "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'model_type': self.best_model_name,
            'accuracy': float(self.best_accuracy),
            'classes': self.label_encoder.classes_.tolist(),
            'feature_dimension': int(self.X.shape[1]),
            'training_samples': int(len(self.X)),
            'training_date': datetime.now().isoformat(),
            'model_file': str(model_path.name),
            'encoder_file': 'label_encoder.pkl'
        }
        
        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\\n✅ MODEL SAVED SUCCESSFULLY!")
        print(f"Model: {model_path}")
        print(f"Encoder: {encoder_path}")
        print(f"Metadata: {metadata_path}")
        print(f"\\nReady for prediction! Run: python landmark_predictor.py")
    
    def plot_confusion_matrix(self):
        """Generate and save confusion matrix plot"""
        if self.best_model is None:
            return
        
        # Generate predictions for confusion matrix
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        if self.best_model_name == 'deep_neural_network':
            y_pred_proba = self.best_model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = self.best_model.predict(X_test)
        
        # Convert to letter labels
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=self.label_encoder.classes_)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_, 
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'ASL Landmark Classification\\nConfusion Matrix ({self.best_model_name})')
        plt.xlabel('Predicted Letter')
        plt.ylabel('True Letter')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved: {plot_path}")
    
    def train_all_models(self):
        """Train all available models and select the best one"""
        print("Starting comprehensive model training...")
        print("This may take several minutes...")
        
        # Load data
        self.load_data()
        
        # Train models
        self.train_random_forest()
        self.train_svm()
        self.train_neural_network_sklearn()
        
        if TENSORFLOW_AVAILABLE:
            self.train_deep_neural_network()
        
        # Evaluate and select best
        self.evaluate_models()
        
        # Generate visualizations
        self.plot_confusion_matrix()
        
        # Save best model
        self.save_best_model()

def main():
    print("ASL Landmark Model Trainer")
    print("This will train multiple ML models and select the best performer.")
    
    trainer = LandmarkModelTrainer()
    trainer.train_all_models()
    
    print("\\n🎉 Training Complete!")
    print("Your ASL landmark recognition model is ready to use.")

if __name__ == "__main__":
    main()