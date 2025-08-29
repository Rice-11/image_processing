#!/usr/bin/env python3
"""
Debug camera access issues
"""

import cv2
import time

def debug_camera():
    print("🔍 Debugging camera access...")
    
    # Try different camera indices
    for i in range(5):
        print(f"\n📷 Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"✅ Camera {i} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ Frame read successfully: {width}x{height}")
                print(f"🎯 Camera {i} is working!")
                
                # Show preview for 3 seconds
                print("Showing 3-second preview...")
                start_time = time.time()
                
                while time.time() - start_time < 3:
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow(f'Camera {i} Test', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                cv2.destroyAllWindows()
                cap.release()
                return i  # Return working camera index
                
            else:
                print(f"❌ Camera {i} opened but can't read frames")
        else:
            print(f"❌ Camera {i} failed to open")
        
        cap.release()
    
    print("\n❌ No working cameras found")
    print("\n💡 Troubleshooting steps:")
    print("1. Grant camera permission in System Preferences")
    print("2. Close other apps using the camera")
    print("3. Try restarting Terminal")
    print("4. Check if external camera is properly connected")
    
    return None

if __name__ == "__main__":
    working_camera = debug_camera()
    
    if working_camera is not None:
        print(f"\n🎉 Use camera index {working_camera} for the sign language interpreter:")
        print(f"python3 main.py --predict --camera {working_camera}")
    else:
        print("\n⚠️  Please fix camera access first")