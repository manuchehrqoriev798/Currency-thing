#!/usr/bin/env python3
"""
test_all.py - Test currency detection on multiple images.

Processes all images from the 'sample' folder and saves annotated results
to the 'sample_result' folder with bounding boxes, labels, and confidence levels.
"""

import cv2
import os
import sys
import glob
from roboflow import Roboflow

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
    return mapping.get(roboflow_class.strip(), roboflow_class)


def calculate_overlap(det1, det2):
    """Calculate IoU (Intersection over Union) between two detections."""
    x1_1 = det1['x'] - det1['width'] / 2
    y1_1 = det1['y'] - det1['height'] / 2
    x2_1 = det1['x'] + det1['width'] / 2
    y2_1 = det1['y'] + det1['height'] / 2
    
    x1_2 = det2['x'] - det2['width'] / 2
    y1_2 = det2['y'] - det2['height'] / 2
    x2_2 = det2['x'] + det2['width'] / 2
    y2_2 = det2['y'] + det2['height'] / 2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = det1['width'] * det1['height']
    area2 = det2['width'] * det2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def test_image(image_path: str, model):
    """
    Test currency detection on a single image.
    
    Args:
        image_path: Path to the image file
        model: Roboflow model object
    
    Returns:
        Tuple of (list of detections, image)
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"  ❌ Image not found: {image_path}")
        return [], None
    
    # Load image - NO RESIZING, use original size
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"  ❌ Could not load image: {image_path}")
        return [], None
    
    height, width = original_image.shape[:2]
    print(f"  ✓ Image size: {width}x{height} (using original size, no resizing)")
    
    # Convert BGR to RGB for Roboflow - use original image
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Run detection - USE ROBOFLOW'S DEFAULT SETTINGS (no custom thresholds)
    # Just use whatever Roboflow provides directly
    try:
        result = model.predict(image_rgb)
        predictions = result.json()
        detections = predictions.get('predictions', [])
        
        print(f"  🔍 Roboflow detected {len(detections)} object(s)")
        
        # Show what Roboflow detected (exactly as provided)
        if detections:
            for idx, det in enumerate(detections, 1):
                det_class = det.get('class', 'Unknown')
                det_conf = det.get('confidence', 0.0)
                print(f"     {idx}. Class: '{det_class}' - Confidence: {det_conf:.1%}")
        
        # Return EXACTLY what Roboflow provides - no filtering, no processing
        return detections, original_image
        
    except Exception as e:
        print(f"  ❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        return [], original_image


def main():
    """Main function."""
    # Get API credentials
    api_key = os.getenv('ROBOFLOW_API_KEY', 'OR7LrKLkb5DbaRn95f5X')
    workspace = os.getenv('ROBOFLOW_WORKSPACE', 'trial-ilic7')
    project = os.getenv('ROBOFLOW_PROJECT', 'main-fddlv')  # Project ID: main-fddlv
    model_version = int(os.getenv('ROBOFLOW_MODEL_VERSION', '2'))  # Using version 2
    
    print("=" * 70)
    print("🧪 Testing Currency Detection on Multiple Images")
    print("=" * 70)
    print(f"Workspace: {workspace}")
    print(f"Project: {project}")
    print(f"Model Version: {model_version}")
    print()
    
    # Connect to Roboflow
    print("🔌 Connecting to Roboflow...")
    try:
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        
        # Try to find an available model version
        model = None
        versions_to_try = [model_version] + [v for v in range(1, 11) if v != model_version]
        
        for v in versions_to_try:
            try:
                version_obj = project_obj.version(v)
                test_model = version_obj.model
                if test_model is not None:
                    model = test_model
                    print(f"✓ Connected! Using model version {v}\n")
                    break
            except Exception:
                continue
        
        if model is None:
            print(f"❌ No trained models found")
            print(f"   Make sure your model is trained and deployed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error connecting to Roboflow: {e}")
        sys.exit(1)
    
    # Get all images from 'sample' folder
    sample_folder = 'sample'
    result_folder = 'sample_result'
    
    # Create result folder if it doesn't exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"📁 Created output folder: {result_folder}")
    
    # Check if sample folder exists
    if not os.path.exists(sample_folder):
        print(f"❌ Error: '{sample_folder}' folder not found!")
        print(f"   Please create a '{sample_folder}' folder and add your images there.")
        sys.exit(1)
    
    # Find all image files in sample folder (common formats)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(sample_folder, ext)))
    
    # Sort images for consistent processing order
    test_images.sort()
    
    if not test_images:
        print(f"❌ No images found in '{sample_folder}' folder!")
        print(f"   Supported formats: JPG, JPEG, PNG")
        sys.exit(1)
    
    print(f"📸 Found {len(test_images)} image(s) in '{sample_folder}' folder")
    print(f"📁 Results will be saved to '{result_folder}' folder")
    print()
    
    print("=" * 70)
    print("📊 Test Results")
    print("=" * 70)
    print()
    
    all_results = {}
    
    for img_path in test_images:
        img_filename = os.path.basename(img_path)
        print(f"🔍 Testing: {img_filename}")
        detections, image = test_image(img_path, model)
        all_results[img_path] = detections
        
        # Prepare output image (always save, even if no detections)
        if image is not None:
            output_image = image.copy()
            
            if not detections:
                print(f"  ❌ No detections from Roboflow")
            else:
                print(f"  ✅ Found {len(detections)} detection(s):")
                for i, det in enumerate(detections, 1):
                    # Use EXACT class name from Roboflow (no swap)
                    class_name = det.get('class', 'Unknown')
                    confidence = det.get('confidence', 0.0)
                    x = int(det['x'] - det['width'] / 2)
                    y = int(det['y'] - det['height'] / 2)
                    w = int(det['width'])
                    h = int(det['height'])
                    print(f"     {i}. {class_name} - Confidence: {confidence:.1%} - Position: ({x}, {y}) to ({x+w}, {y+h})")
                
                # Draw annotations on image - use EXACT class names from Roboflow
                # Color mapping based on Roboflow's class names
                colors = {
                    '20': (0, 0, 255),      # Red for 20
                    '100': (0, 255, 0),     # Green for 100
                }
                color_names = {
                    '20': 'Red',
                    '100': 'Green',
                }
                
                for i, det in enumerate(detections, 1):
                    # Get bounding box coordinates and ensure they're within image bounds
                    img_height, img_width = output_image.shape[:2]
                    x = int(det['x'] - det['width'] / 2)
                    y = int(det['y'] - det['height'] / 2)
                    w = int(det['width'])
                    h = int(det['height'])
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, min(x, img_width - 1))
                    y = max(0, min(y, img_height - 1))
                    w = min(w, img_width - x)
                    h = min(h, img_height - y)
                    
                    # Use EXACT class name from Roboflow (no swap, no modification)
                    class_name = det.get('class', 'Unknown')
                    confidence = det.get('confidence', 0.0)
                    color = colors.get(class_name, (255, 255, 255))
                    color_name = color_names.get(class_name, 'White')
                    
                    # Draw bounding box with thicker lines for visibility
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 4)
                    
                    # Draw label with EXACT class name from Roboflow
                    label = f"{i}. {class_name} ({color_name}) - {confidence:.1%}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    
                    # Label background
                    cv2.rectangle(output_image, (x, y - label_size[1] - 20), 
                                 (x + label_size[0] + 15, y), color, -1)
                    
                    # Label text
                    cv2.putText(output_image, label, (x + 8, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
                    # Draw corners for better visibility
                    corner_size = 15
                    cv2.line(output_image, (x, y), (x + corner_size, y), color, 3)
                    cv2.line(output_image, (x, y), (x, y + corner_size), color, 3)
                    cv2.line(output_image, (x + w, y), (x + w - corner_size, y), color, 3)
                    cv2.line(output_image, (x + w, y), (x + w, y + corner_size), color, 3)
                    cv2.line(output_image, (x, y + h), (x + corner_size, y + h), color, 3)
                    cv2.line(output_image, (x, y + h), (x, y + h - corner_size), color, 3)
                    cv2.line(output_image, (x + w, y + h), (x + w - corner_size, y + h), color, 3)
                    cv2.line(output_image, (x + w, y + h), (x + w, y + h - corner_size), color, 3)
            
            # Save image to sample_result folder (with or without annotations)
            img_filename = os.path.basename(img_path)
            img_name, img_ext = os.path.splitext(img_filename)
            output_filename = f"{img_name}_result{img_ext}"
            output_path = os.path.join(result_folder, output_filename)
            cv2.imwrite(output_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  📸 Result image saved: {output_path}")
        print()
    
    # Summary
    print("=" * 70)
    print("📋 Summary")
    print("=" * 70)
    for img_path in test_images:
        img_filename = os.path.basename(img_path)
        detections = all_results[img_path]
        if detections:
            # Use EXACT class names from Roboflow (no swap)
            classes = [det.get('class', 'Unknown') for det in detections]
            confidences = [f"{det.get('confidence', 0.0):.1%}" for det in detections]
            print(f"  {img_filename}: {len(detections)} detection(s) - Classes: {', '.join(classes)} (Confidence: {', '.join(confidences)})")
        else:
            print(f"  {img_filename}: No detections")
    print("=" * 70)
    print(f"\n✅ All results saved to '{result_folder}' folder")
    print("=" * 70)


if __name__ == "__main__":
    main()

