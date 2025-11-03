"""
Setup verification script
Checks if all requirements are met before running the detection system
"""

import os
import sys

def check_files():
    """Check if currency folders exist and contain images"""
    print("Checking currency folder structure...")
    currencies_folder = "currencies"
    required_folders = ['20', '50', '100', '500', '1000', '5000']
    missing_folders = []
    empty_folders = []
    image_extensions = ['.jpeg', '.jpg', '.png', '.bmp']
    total_images = 0
    
    if not os.path.exists(currencies_folder):
        print(f"  ✗ '{currencies_folder}' folder NOT FOUND")
        print(f"    Please create the '{currencies_folder}' folder with subfolders for each denomination")
        return False, []
    
    if not os.path.isdir(currencies_folder):
        print(f"  ✗ '{currencies_folder}' is not a directory")
        return False, []
    
    print(f"  ✓ '{currencies_folder}' folder found")
    
    for folder_name in required_folders:
        folder_path = os.path.join(currencies_folder, folder_name)
        if not os.path.exists(folder_path):
            print(f"  ✗ Folder '{folder_name}' MISSING in {currencies_folder}/")
            missing_folders.append(folder_name)
        elif not os.path.isdir(folder_path):
            print(f"  ✗ '{folder_path}' is not a directory")
            missing_folders.append(folder_name)
        else:
            # Count images in folder
            files = os.listdir(folder_path)
            image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
            image_count = len(image_files)
            
            if image_count == 0:
                print(f"  ⚠ Folder '{folder_name}' exists but contains NO images")
                empty_folders.append(folder_name)
            else:
                print(f"  ✓ Folder '{folder_name}' contains {image_count} image(s)")
                total_images += image_count
    
    if missing_folders or empty_folders:
        return False, missing_folders + empty_folders
    
    print(f"\n  Total images found: {total_images}")
    return True, []

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking Python dependencies...")
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package} installed")
        except ImportError:
            print(f"  ✗ {package} NOT INSTALLED")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_camera():
    """Check if camera is available"""
    print("\nChecking camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("  ✓ Camera is accessible")
            cap.release()
            return True
        else:
            print("  ✗ Camera could not be opened")
            return False
    except Exception as e:
        print(f"  ✗ Error checking camera: {e}")
        return False

def main():
    print("=" * 60)
    print("Currency Detection System - Setup Verification")
    print("=" * 60)
    
    # Check files
    files_ok, missing_files = check_files()
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    
    # Check camera
    camera_ok = check_camera()
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    all_ok = files_ok and deps_ok and camera_ok
    
    if all_ok:
        print("✓ All checks passed! You can run currency_detector.py")
    else:
        print("✗ Some checks failed:")
        if not files_ok:
            if missing_files:
                print(f"  - Missing or empty folders in 'currencies/': {', '.join(missing_files)}")
                print("    Folder structure should be:")
                print("      currencies/")
                print("        ├── 20/")
                print("        ├── 50/")
                print("        ├── 100/")
                print("        ├── 500/")
                print("        ├── 1000/")
                print("        └── 5000/")
                print("    Add image files (.jpeg, .jpg, .png, .bmp) to each folder")
        if not deps_ok:
            print(f"  - Missing packages: {', '.join(missing_deps)}")
            print("    Run: pip install -r requirements.txt")
        if not camera_ok:
            print("  - Camera not accessible")
            print("    Make sure your camera is connected and not in use by another app")
    
    print("=" * 60)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

