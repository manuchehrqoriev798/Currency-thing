#!/usr/bin/env python3
"""
Helper script to list all available cameras and help set up phone camera.
"""

import cv2
import sys

def list_all_cameras():
    """List all available cameras with details."""
    print("=" * 70)
    print("CAMERA DETECTION - Listing all available cameras")
    print("=" * 70)
    print()
    
    cameras = []
    for i in range(20):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                
                cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend
                })
                
                print(f"Camera {i}:")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")
                print(f"  Backend: {backend}")
                print()
            cap.release()
    
    if not cameras:
        print("❌ No cameras found!")
        print()
        print("SETUP INSTRUCTIONS FOR PHONE CAMERA:")
        print("-" * 70)
        print()
        print("For Android phones:")
        print("  1. Install 'DroidCam' or 'iVCam' from Play Store")
        print("  2. Connect phone via USB")
        print("  3. Enable USB debugging (Settings > Developer Options)")
        print("  4. Open the webcam app on your phone")
        print("  5. Run this script again to detect the camera")
        print()
        print("For iPhone (Mac only):")
        print("  1. Use Continuity Camera (built-in, no app needed)")
        print("  2. Connect iPhone via USB or use wirelessly")
        print("  3. Camera should appear automatically")
        print()
        print("Alternative: Use wireless connection")
        print("  - DroidCam/iVCam also support WiFi connection")
        print("  - Make sure phone and computer are on same network")
        print()
        return None
    
    print(f"✓ Found {len(cameras)} camera(s)")
    print()
    
    if len(cameras) > 1:
        print("To use a specific camera, run:")
        print(f"  python main.py <camera_index>")
        print()
        print("Or set environment variable:")
        print(f"  export CAMERA_INDEX=<camera_index>")
        print(f"  python main.py")
        print()
        print("Recommended: Use camera index", cameras[-1]['index'], 
              "(usually the phone camera is added last)")
    else:
        print("To use this camera, just run:")
        print("  python main.py")
        print()
    
    return cameras

if __name__ == "__main__":
    cameras = list_all_cameras()
    if cameras:
        sys.exit(0)
    else:
        sys.exit(1)



