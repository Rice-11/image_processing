import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import urllib.request
from pathlib import Path

class SignLanguageDataLoader:
    def __init__(self):
        self.class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]
        self.num_classes = len(self.class_names)
        self.img_size = 28
        
    def download_dataset(self):
        """Download the Sign Language MNIST dataset"""
        print("Downloading Sign Language MNIST dataset...")
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        train_file = data_dir / "sign_mnist_train.csv"
        test_file = data_dir / "sign_mnist_test.csv"
        
        # Check if files already exist
        if train_file.exists() and test_file.exists():
            print("Dataset files found locally, loading...")
            try:
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)
                print(f"Training data shape: {train_df.shape}")
                print(f"Test data shape: {test_df.shape}")
                return train_df, test_df
            except Exception as e:
                print(f"Error loading local files: {e}")
                print("Attempting to download fresh copies...")
        
        # Download URLs for the Sign Language MNIST dataset
        train_url = "https://www.dropbox.com/s/tkxumwl7wgs2b6z/sign_mnist_train.csv?dl=1"
        test_url = "https://www.dropbox.com/s/8ucdgqs8kqns3f0/sign_mnist_test.csv?dl=1"
        
        try:
            # Download training data
            print("Downloading training data...")
            urllib.request.urlretrieve(train_url, train_file)
            
            # Download test data
            print("Downloading test data...")  
            urllib.request.urlretrieve(test_url, test_file)
            
            # Load the datasets
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            print(f"Training data shape: {train_df.shape}")
            print(f"Test data shape: {test_df.shape}")
            print("Dataset downloaded successfully!")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Trying alternative method...")
            
            # Alternative: Create sample dataset for testing
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self):
        """Create a small sample dataset for testing purposes"""
        print("Creating sample dataset for testing...")
        
        # Create small sample data (just for demo purposes)
        np.random.seed(42)
        
        # Generate sample training data
        n_train_samples = 1000
        n_test_samples = 200
        
        # Random 28x28 images
        X_train_flat = np.random.randint(0, 255, (n_train_samples, 784))
        y_train = np.random.randint(0, 24, n_train_samples)
        
        X_test_flat = np.random.randint(0, 255, (n_test_samples, 784))
        y_test = np.random.randint(0, 24, n_test_samples)
        
        # Create DataFrames
        train_columns = ['label'] + [f'pixel{i}' for i in range(784)]
        test_columns = ['label'] + [f'pixel{i}' for i in range(784)]
        
        train_data = np.column_stack([y_train, X_train_flat])
        test_data = np.column_stack([y_test, X_test_flat])
        
        train_df = pd.DataFrame(train_data, columns=train_columns)
        test_df = pd.DataFrame(test_data, columns=test_columns)
        
        print("⚠️  Using sample dataset for demonstration")
        print("For real ASL recognition, please download the actual dataset")
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df
    
    def preprocess_data(self, train_df, test_df, validation_split=0.2):
        """Preprocess the dataset for training"""
        print("Preprocessing data...")
        
        # Extract features and labels
        X_train = train_df.drop('label', axis=1).values
        y_train = train_df['label'].values
        
        X_test = test_df.drop('label', axis=1).values  
        y_test = test_df['label'].values
        
        # Reshape to image format (28x28)
        X_train = X_train.reshape(-1, self.img_size, self.img_size, 1)
        X_test = X_test.reshape(-1, self.img_size, self.img_size, 1)
        
        # Normalize pixel values
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        
        # Create validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=validation_split, 
            random_state=42, 
            stratify=y_train
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def visualize_samples(self, X_data, y_data, num_samples=12):
        """Visualize sample images from the dataset"""
        plt.figure(figsize=(12, 8))
        
        for i in range(min(num_samples, len(X_data))):
            plt.subplot(3, 4, i + 1)
            plt.imshow(X_data[i].reshape(28, 28), cmap='gray')
            label_idx = np.argmax(y_data[i])
            plt.title(f'Label: {self.class_names[label_idx]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
    
    def preprocess_camera_frame(self, frame):
        """Preprocess camera frame for prediction"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        resized = cv2.resize(gray, (self.img_size, self.img_size))
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        # Reshape for model input
        preprocessed = normalized.reshape(1, self.img_size, self.img_size, 1)
        
        return preprocessed
    
    def extract_hand_roi(self, frame, roi_size=200):
        """Extract region of interest for hand detection"""
        height, width = frame.shape[:2]
        
        # Calculate ROI coordinates (center of frame)
        x1 = (width - roi_size) // 2
        y1 = (height - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        return roi, (x1, y1, x2, y2)
    
    def get_class_distribution(self, y_data):
        """Get class distribution for analysis"""
        labels = np.argmax(y_data, axis=1)
        unique, counts = np.unique(labels, return_counts=True)
        
        distribution = {}
        for i, count in zip(unique, counts):
            distribution[self.class_names[i]] = count
            
        return distribution
    
    def plot_class_distribution(self, y_data, title="Class Distribution"):
        """Plot class distribution"""
        distribution = self.get_class_distribution(y_data)
        
        plt.figure(figsize=(12, 6))
        plt.bar(distribution.keys(), distribution.values())
        plt.title(title)
        plt.xlabel('Sign Language Letters')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return distribution