#!/usr/bin/env python3
"""
Quick test of the ASL system components
"""

import os
import sys

def test_imports():
    """Test that all components can be imported"""
    try:
        print("Testing imports...")
        
        print("  - Testing data collector...")
        from asl_data_collector import ASLDataCollector
        print("    ✓ Data collector imported successfully")
        
        print("  - Testing model trainer...")
        from asl_model_trainer import ASLModelTrainer
        print("    ✓ Model trainer imported successfully")
        
        print("  - Testing live predictor...")
        from asl_live_predictor import ASLLivePredictor
        print("    ✓ Live predictor imported successfully")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Import error: {e}")
        return False

def test_camera():
    """Test camera access"""
    try:
        import cv2
        print("Testing camera access...")
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("    ✓ Camera is working")
                cap.release()
                return True
            else:
                print("    ✗ Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print("    ✗ Cannot open camera")
            return False
            
    except Exception as e:
        print(f"    ✗ Camera test error: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hand detection"""
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        print("Testing MediaPipe hand detection...")
        
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        # Create a dummy image (simple hand-like shape)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Process the image
        results = hands.process(test_image)
        
        print("    ✓ MediaPipe initialized and can process images")
        hands.close()
        return True
        
    except Exception as e:
        print(f"    ✗ MediaPipe test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("ASL RECOGNITION SYSTEM TEST")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test camera
    if test_camera():
        tests_passed += 1
    
    # Test MediaPipe
    if test_mediapipe():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 50)
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour ASL recognition system is ready!")
        print("\nNext steps:")
        print("1. Run: python asl_data_collector.py")
        print("2. Run: python asl_model_trainer.py") 
        print("3. Run: python asl_live_predictor.py")
    else:
        print("❌ Some tests failed.")
        print("\nPlease check the errors above and:")
        print("1. Ensure all required packages are installed")
        print("2. Check camera permissions") 
        print("3. Verify Python environment")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()