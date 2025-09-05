#!/usr/bin/env python3
"""
Download the real Sign Language MNIST dataset
"""

import os
import requests
import zipfile
import pandas as pd
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"SUCCESS: Downloaded {filename}")
        return True
    else:
        print(f"ERROR: Failed to download {filename} - Status code: {response.status_code}")
        return False

def main():
    print("Downloading Real Sign Language MNIST Dataset")
    print("=" * 50)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # URLs for the real Sign Language MNIST dataset
    urls = {
        "sign_mnist_train.csv": "https://raw.githubusercontent.com/datahub-io/sign-language-mnist/master/sign_mnist_train.csv",
        "sign_mnist_test.csv": "https://raw.githubusercontent.com/datahub-io/sign-language-mnist/master/sign_mnist_test.csv"
    }
    
    # Alternative URLs if the above don't work
    alt_urls = {
        "sign_mnist_train.csv": "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/sign_mnist_train.csv",
        "sign_mnist_test.csv": "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/sign_mnist_test.csv"
    }
    
    success_count = 0
    
    for filename, url in urls.items():
        filepath = data_dir / filename
        
        # Try primary URL first
        if download_file(url, filepath):
            success_count += 1
        else:
            # Try alternative URL
            alt_url = alt_urls.get(filename)
            if alt_url and download_file(alt_url, filepath):
                success_count += 1
            else:
                print(f"FAILED: Could not download {filename} from any source")
    
    print(f"\nDownload Summary:")
    print(f"Successfully downloaded: {success_count}/2 files")
    
    # Verify the downloaded files
    if success_count > 0:
        print("\nVerifying downloaded files...")
        
        for filename in ["sign_mnist_train.csv", "sign_mnist_test.csv"]:
            filepath = data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    print(f"SUCCESS: {filename} - Shape: {df.shape}")
                    
                    if len(df.columns) == 785:  # 1 label + 784 pixels
                        print(f"  ✓ Correct format (785 columns)")
                        print(f"  ✓ Classes: {sorted(df.iloc[:, 0].unique())}")
                        print(f"  ✓ Samples: {len(df)}")
                    else:
                        print(f"  ⚠️ Unexpected format ({len(df.columns)} columns)")
                        
                except Exception as e:
                    print(f"ERROR: Could not read {filename} - {e}")
    
    print("\n" + "=" * 50)
    if success_count == 2:
        print("✅ READY TO RETRAIN!")
        print("Run: python train_sign_mnist_model.py")
    else:
        print("❌ Download incomplete. Please check your internet connection.")

if __name__ == "__main__":
    main()