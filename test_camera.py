#!/usr/bin/env python3
"""
Quick camera test for Sign Language Interpreter setup
"""

import cv2
import sys

def test_camera():
    print("🎥 Testing camera access...")
    
    # Try to access the default camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera access failed!")
        print("\n📱 macOS Users:")
        print("1. Go to System Preferences → Security & Privacy → Privacy → Camera")
        print("2. Enable camera access for Terminal and/or VS Code")
        print("3. Run this test again")
        print("\n🔍 Troubleshooting:")
        print("- Try disconnecting and reconnecting external cameras")
        print("- Restart Terminal/VS Code after granting permissions")
        print("- Check if other apps are using the camera")
        return False
    
    print("✅ Camera access granted!")
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if ret:
        height, width = frame.shape[:2]
        print(f"📏 Camera resolution: {width}x{height}")
        print("🎯 Camera is ready for sign language recognition!")
        
        # Show a quick preview (optional)
        print("\nPress 'q' to close camera preview...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.imshow('Camera Test - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        success = True
    else:
        print("❌ Could not read from camera")
        success = False
    
    cap.release()
    return success

if __name__ == "__main__":
    if test_camera():
        print("\n🎉 Setup complete! Camera is working properly.")
        print("You can now run: python3 main.py --demo")
    else:
        print("\n⚠️  Please fix camera access and try again.")
        sys.exit(1)