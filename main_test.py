#!/usr/bin/env python3
"""
main_test.py - Test currency detection using Roboflow Inference API.

Sends images from 'sample' folder to Roboflow and prints the results.
Uses the proper Inference API to match web interface results.
"""

import os
import sys
import glob
from inference import get_model
import cv2

# Configuration
MODEL_ID = "main-fddlv/2"  # Model ID from Roboflow dashboard
SAMPLE_FOLDER = "sample"

# Make sure ROBOFLOW_API_KEY is set in environment
# export ROBOFLOW_API_KEY=OR7LrKLkb5DbaRn95f5X


def map_class_name(roboflow_class: str) -> str:
    """
    Map Roboflow class names to display names.
    IMPORTANT: When Roboflow detects "20", display "100", and vice versa.
    
    Args:
        roboflow_class: Class name from Roboflow detection
    
    Returns:
        Mapped class name for display
    """
    mapping = {
        '20': '100',   # Roboflow detects "20" -> display "100"
        '100': '20',   # Roboflow detects "100" -> display "20"
    }
    return mapping.get(str(roboflow_class).strip(), roboflow_class)


def process_image(image_path: str, model):
    """
    Send image to Roboflow and get results.
    
    Args:
        image_path: Path to the image file
        model: Roboflow inference model object
    
    Returns:
        Results from Roboflow
    """
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"  ❌ Image not found")
        return None
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ❌ Could not load image")
        return None
    
    try:
        # Run inference - this matches what the web interface does
        results = model.infer(image)[0]
        
        # Access predictions from the response object (it's a Pydantic model, not a dict)
        predictions = results.predictions if hasattr(results, 'predictions') else []
        
        print(f"  ✅ Roboflow returned {len(predictions)} detection(s):")
        
        for idx, pred in enumerate(predictions, 1):
            # Access attributes directly from the prediction object
            roboflow_class = pred.class_name if hasattr(pred, 'class_name') else (pred.class_ if hasattr(pred, 'class_') else 'Unknown')
            pred_class = map_class_name(roboflow_class)  # Apply swap mapping
            pred_conf = pred.confidence if hasattr(pred, 'confidence') else 0.0
            x = pred.x if hasattr(pred, 'x') else 0
            y = pred.y if hasattr(pred, 'y') else 0
            w = pred.width if hasattr(pred, 'width') else 0
            h = pred.height if hasattr(pred, 'height') else 0
            class_id = pred.class_id if hasattr(pred, 'class_id') else 'N/A'
            detection_id = pred.detection_id if hasattr(pred, 'detection_id') else 'N/A'
            
            print(f"     {idx}. Class: '{pred_class}' (ID: {class_id})")
            print(f"         Confidence: {pred_conf:.1%}")
            print(f"         Position: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
            print(f"         Detection ID: {detection_id}")
        
        # Convert to dict for summary (or return the object)
        return results
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""
    print("=" * 70)
    print("🧪 Testing Currency Detection - Roboflow Inference API")
    print("=" * 70)
    print(f"Model ID: {MODEL_ID}")
    print()
    
    # Check API key
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        print("❌ Error: ROBOFLOW_API_KEY environment variable not set!")
        print("   Please run: export ROBOFLOW_API_KEY=OR7LrKLkb5DbaRn95f5X")
        sys.exit(1)
    
    # Check if sample folder exists
    if not os.path.exists(SAMPLE_FOLDER):
        print(f"❌ Error: '{SAMPLE_FOLDER}' folder not found!")
        sys.exit(1)
    
    # Find all image files in sample folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(SAMPLE_FOLDER, ext)))
    
    test_images.sort()
    
    if not test_images:
        print(f"❌ No images found in '{SAMPLE_FOLDER}' folder!")
        sys.exit(1)
    
    print(f"📸 Found {len(test_images)} image(s)")
    print()
    
    # Load model using Inference API
    print("🔌 Loading model from Roboflow...")
    try:
        model = get_model(model_id=MODEL_ID)
        print(f"✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"   Make sure:")
        print(f"   1. ROBOFLOW_API_KEY is set correctly")
        print(f"   2. Model ID '{MODEL_ID}' is correct")
        print(f"   3. You have access to this model")
        sys.exit(1)
    
    print("=" * 70)
    print("📊 Results from Roboflow")
    print("=" * 70)
    
    all_results = {}
    
    # Process each image
    for img_path in test_images:
        results = process_image(img_path, model)
        all_results[img_path] = results
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 Summary")
    print("=" * 70)
    for img_path in test_images:
        img_filename = os.path.basename(img_path)
        results = all_results[img_path]
        if results:
            predictions = results.predictions if hasattr(results, 'predictions') else []
            if predictions:
                classes = []
                for pred in predictions:
                    roboflow_class = pred.class_name if hasattr(pred, 'class_name') else (pred.class_ if hasattr(pred, 'class_') else 'Unknown')
                    pred_class = map_class_name(roboflow_class)  # Apply swap mapping
                    classes.append(pred_class)
                print(f"  {img_filename}: {len(predictions)} detection(s) - {', '.join(classes)}")
            else:
                print(f"  {img_filename}: No detections")
        else:
            print(f"  {img_filename}: Failed to process")
    print("=" * 70)


if __name__ == "__main__":
    main()
