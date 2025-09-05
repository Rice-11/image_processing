#!/usr/bin/env python3
"""
Sign Language MNIST Dataset Downloader
Downloads the official Sign Language MNIST dataset for ASL recognition training
"""

import os
import urllib.request
import pandas as pd
import numpy as np

def download_sign_mnist():
    """Download Sign Language MNIST dataset"""
    print("Sign Language MNIST Dataset Downloader")
    print("="*50)
    
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Dataset URLs (from Kaggle/official sources)
    urls = {
        'train': 'https://www.dropbox.com/s/tkxumwl7wgs2b6z/sign_mnist_train.csv?dl=1',
        'test': 'https://www.dropbox.com/s/8ucdgqs8kqns3f0/sign_mnist_test.csv?dl=1'
    }
    
    for dataset_type, url in urls.items():
        filename = f"sign_mnist_{dataset_type}.csv"
        filepath = os.path.join(data_dir, filename)
        
        # Check if file already exists and is valid
        if os.path.exists(filepath):
            try:
                # Try to read a few rows to validate
                test_df = pd.read_csv(filepath, nrows=5)
                if test_df.shape[1] == 785:  # 1 label + 784 pixels
                    print(f"✅ {filename} already exists and appears valid")
                    continue
                else:
                    print(f"⚠️ {filename} exists but appears invalid, re-downloading...")
            except:
                print(f"⚠️ {filename} exists but is corrupted, re-downloading...")
        
        print(f"📥 Downloading {filename}...")
        
        try:
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = (block_num * block_size) / total_size * 100
                    print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, filepath, reporthook=show_progress)
            print()  # New line after progress
            
            # Validate downloaded file
            df = pd.read_csv(filepath, nrows=10)
            if df.shape[1] == 785:
                print(f"✅ {filename} downloaded and validated successfully")
                print(f"   Shape: {df.shape[0]}+ rows × {df.shape[1]} columns")
                print(f"   Labels: {sorted(df.iloc[:, 0].unique())}")
            else:
                print(f"❌ {filename} downloaded but validation failed")
                print(f"   Expected 785 columns, got {df.shape[1]}")
                
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            print("You may need to download manually from:")
            print(f"   {url}")
    
    print("\n" + "="*50)
    print("Download complete!")
    print("You can now run: python train_sign_mnist_model.py")

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    print("\nCreating sample dataset for testing...")
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create synthetic data that mimics Sign Language MNIST format
    np.random.seed(42)
    
    # Generate sample data for 24 classes (A-Y excluding J,Z)
    num_samples_per_class = 50
    total_samples = 24 * num_samples_per_class
    
    # Generate labels (0-23 for A-Y excluding J,Z)
    labels = np.repeat(range(24), num_samples_per_class)
    
    # Generate synthetic 28x28 images (784 pixels)
    # Create different patterns for each class
    images = []
    
    for class_idx in range(24):
        for sample_idx in range(num_samples_per_class):
            # Create a simple pattern for each class
            img = np.zeros((28, 28))
            
            # Add some class-specific patterns
            img[5:23, 5:23] = class_idx * 10  # Base brightness
            img[10:18, 10:18] = 255  # Center square
            
            # Add some noise
            noise = np.random.randint(0, 50, (28, 28))
            img = np.clip(img + noise, 0, 255)
            
            images.append(img.flatten())
    
    images = np.array(images)
    
    # Combine labels and images
    data = np.column_stack([labels, images])
    
    # Create DataFrame
    columns = ['label'] + [f'pixel{i}' for i in range(784)]
    df = pd.DataFrame(data, columns=columns)
    
    # Save as CSV
    sample_path = os.path.join(data_dir, 'sign_mnist_train.csv')
    df.to_csv(sample_path, index=False)
    
    print(f"✅ Sample dataset created: {sample_path}")
    print(f"   Shape: {df.shape}")
    print(f"   This is for TESTING ONLY - use real dataset for actual training")
    
    return sample_path

def main():
    """Main function"""
    print("Choose an option:")
    print("1. Download real Sign Language MNIST dataset (recommended)")
    print("2. Create sample dataset for testing")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        download_sign_mnist()
    elif choice == '2':
        create_sample_dataset()
        print("\n⚠️ NOTE: Sample dataset is for testing only!")
        print("For real ASL recognition, use option 1 to download the actual dataset")
    else:
        print("Invalid choice. Please run again and choose 1 or 2.")

if __name__ == "__main__":
    main()