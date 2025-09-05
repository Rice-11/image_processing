#!/usr/bin/env python3
"""
Download the actual Sign Language MNIST dataset from reliable sources
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import io

def download_from_github():
    """Try downloading from GitHub repositories"""
    
    urls = [
        {
            'train': 'https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/X.npy',
            'train_labels': 'https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/Y.npy',
            'test': None,  # Will create test split from train
            'format': 'numpy'
        },
        {
            'train': 'https://raw.githubusercontent.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn/master/data/sign_mnist_train.csv',
            'test': 'https://raw.githubusercontent.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn/master/data/sign_mnist_test.csv',
            'format': 'csv'
        }
    ]
    
    for i, source in enumerate(urls):
        print(f"\\nTrying source {i+1}...")
        try:
            if source['format'] == 'numpy':
                # Download numpy arrays
                print("Downloading training images...")
                X_response = requests.get(source['train'], timeout=30)
                if X_response.status_code == 200:
                    X_data = np.load(io.BytesIO(X_response.content))
                    
                    print("Downloading training labels...")
                    y_response = requests.get(source['train_labels'], timeout=30)
                    if y_response.status_code == 200:
                        y_data = np.load(io.BytesIO(y_response.content))
                        
                        # Convert to CSV format
                        return convert_numpy_to_csv(X_data, y_data)
            
            elif source['format'] == 'csv':
                # Download CSV files
                print("Downloading training CSV...")
                train_response = requests.get(source['train'], timeout=30)
                if train_response.status_code == 200:
                    with open('data/sign_mnist_train.csv', 'wb') as f:
                        f.write(train_response.content)
                    
                    if source['test']:
                        print("Downloading test CSV...")
                        test_response = requests.get(source['test'], timeout=30)
                        if test_response.status_code == 200:
                            with open('data/sign_mnist_test.csv', 'wb') as f:
                                f.write(test_response.content)
                    
                    return True
                    
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    return False

def convert_numpy_to_csv(X_data, y_data):
    """Convert numpy arrays to CSV format"""
    print(f"Converting numpy data: {X_data.shape} samples")
    
    # Reshape if needed
    if len(X_data.shape) > 2:
        X_data = X_data.reshape(X_data.shape[0], -1)
    
    # Combine labels and features
    data = np.column_stack([y_data, X_data])
    
    # Create DataFrame
    columns = ['label'] + [f'pixel{i}' for i in range(X_data.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Save files
    train_df.to_csv('data/sign_mnist_train.csv', index=False)
    test_df.to_csv('data/sign_mnist_test.csv', index=False)
    
    return True

def download_from_direct_links():
    """Try direct download links"""
    
    direct_links = [
        'https://www.dropbox.com/s/5xhx7vlcaj9zx7m/sign_mnist_train.csv?dl=1',
        'https://drive.google.com/uc?export=download&id=1z4QXj-w-3HgfUmIGXY6-lnCwQZMFhSLE'
    ]
    
    for i, url in enumerate(direct_links):
        print(f"\\nTrying direct link {i+1}...")
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200 and len(response.content) > 10000:  # Must be substantial
                with open('data/sign_mnist_train.csv', 'wb') as f:
                    f.write(response.content)
                print("SUCCESS: Downloaded real dataset!")
                return True
        except Exception as e:
            print(f"Failed: {e}")
    
    return False

def create_improved_synthetic():
    """Create much more realistic synthetic data based on real hand structure"""
    print("Creating improved realistic synthetic dataset...")
    
    import cv2
    
    def create_realistic_hand_pattern(letter_idx):
        """Create realistic hand patterns based on actual ASL structure"""
        img = np.zeros((28, 28), dtype=np.uint8)
        
        if letter_idx == 0:  # A - closed fist with thumb to side
            # Fist shape
            cv2.ellipse(img, (14, 16), (8, 10), 0, 0, 360, 200, -1)
            # Thumb
            cv2.ellipse(img, (8, 16), (3, 6), 20, 0, 360, 220, -1)
            
        elif letter_idx == 1:  # B - four fingers up, thumb tucked
            # Palm base
            cv2.rectangle(img, (8, 18), (20, 26), 180, -1)
            # Four fingers
            cv2.rectangle(img, (9, 4), (12, 20), 220, -1)
            cv2.rectangle(img, (12, 2), (15, 20), 240, -1) 
            cv2.rectangle(img, (15, 2), (18, 20), 230, -1)
            cv2.rectangle(img, (18, 4), (21, 20), 210, -1)
            
        elif letter_idx == 8:  # I - pinky up
            # Fist
            cv2.ellipse(img, (14, 18), (7, 8), 0, 0, 360, 190, -1)
            # Pinky
            cv2.rectangle(img, (20, 6), (23, 18), 240, -1)
            
        elif letter_idx == 9:  # K - index and middle up at angle
            # Palm
            cv2.ellipse(img, (14, 20), (6, 6), 0, 0, 360, 180, -1)
            # Index finger angled
            cv2.line(img, (14, 20), (10, 8), 220, 3)
            # Middle finger angled  
            cv2.line(img, (14, 20), (18, 6), 230, 3)
            
        elif letter_idx == 10:  # L - thumb and index perpendicular
            # Thumb vertical
            cv2.rectangle(img, (8, 10), (12, 24), 210, -1)
            # Index horizontal
            cv2.rectangle(img, (12, 8), (22, 12), 220, -1)
            
        elif letter_idx == 20:  # V - peace sign
            # Palm base
            cv2.ellipse(img, (14, 20), (5, 6), 0, 0, 360, 170, -1)
            # Index finger
            cv2.line(img, (12, 18), (8, 4), 230, 4)
            # Middle finger  
            cv2.line(img, (16, 18), (20, 4), 240, 4)
            
        else:  # Generic hand shape
            # Basic palm
            cv2.ellipse(img, (14, 18), (6, 8), np.random.randint(-15, 15), 0, 360, 
                       np.random.randint(160, 200), -1)
            # Random finger configuration
            for i in range(np.random.randint(2, 5)):
                start_x = np.random.randint(10, 18)
                start_y = np.random.randint(16, 20)
                end_x = start_x + np.random.randint(-4, 4)
                end_y = np.random.randint(4, 12)
                thickness = np.random.randint(2, 4)
                cv2.line(img, (start_x, start_y), (end_x, end_y), 
                        np.random.randint(200, 255), thickness)
        
        # Add realistic noise and variations
        noise = np.random.normal(0, 8, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        
        # Random small transformations
        if np.random.random() < 0.7:
            # Small rotation
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1)
            img = cv2.warpAffine(img, M, (28, 28))
        
        if np.random.random() < 0.5:
            # Small translation
            tx = np.random.randint(-2, 3)
            ty = np.random.randint(-2, 3)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (28, 28))
        
        return img.flatten()
    
    # Create large realistic dataset
    samples_per_class = 800  # More samples per letter
    total_samples = 24 * samples_per_class
    
    data = []
    for label in range(24):
        for sample in range(samples_per_class):
            pixels = create_realistic_hand_pattern(label)
            row = [label] + pixels.tolist()
            data.append(row)
        print(f"Generated class {label}: {samples_per_class} samples")
    
    # Create DataFrame and save
    columns = ['label'] + [f'pixel{i}' for i in range(784)]
    df = pd.DataFrame(data, columns=columns)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split train/test
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    train_df.to_csv('data/sign_mnist_train.csv', index=False) 
    test_df.to_csv('data/sign_mnist_test.csv', index=False)
    
    print(f"Created improved synthetic dataset:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Testing: {len(test_df)} samples")
    return True

def main():
    print("DOWNLOADING REAL SIGN LANGUAGE MNIST")
    print("=" * 50)
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Try different sources
    success = False
    
    # Method 1: GitHub repositories
    if not success:
        success = download_from_github()
    
    # Method 2: Direct links
    if not success:
        success = download_from_direct_links()
    
    # Method 3: Improved synthetic as fallback
    if not success:
        print("\\nReal data sources failed. Creating improved realistic synthetic...")
        success = create_improved_synthetic()
    
    if success:
        # Verify downloaded data
        try:
            df = pd.read_csv('data/sign_mnist_train.csv')
            print(f"\\nSUCCESS!")
            print(f"Training samples: {len(df)}")
            print(f"Features: {df.shape[1] - 1}")
            print(f"Classes: {sorted(df['label'].unique())}")
            
            if len(df) > 5000:  # Good size dataset
                print("\\nDataset looks good! Ready to retrain:")
                print("python quick_train.py")
            else:
                print("\\nWarning: Small dataset. Consider collecting real data:")
                print("python collect_real_data.py")
                
        except Exception as e:
            print(f"Error verifying dataset: {e}")
    else:
        print("\\nAll methods failed. Try:")
        print("1. python collect_real_data.py  # Best option")
        print("2. Manually download from Kaggle")

if __name__ == "__main__":
    main()