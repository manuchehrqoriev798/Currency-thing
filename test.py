#!/usr/bin/env python3
"""
test.py - Test currency detection on a single image.

Usage:
    python3 test.py 100.jpg
    python3 test.py 100.jpg
"""

import cv2
import os
import sys
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


def test_image(image_path: str):
    """
    Test currency detection on a single image.
    
    Args:
        image_path: Path to the image file
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Get API credentials
    api_key = os.getenv('ROBOFLOW_API_KEY', 'OR7LrKLkb5DbaRn95f5X')
    workspace = os.getenv('ROBOFLOW_WORKSPACE', 'trial-ilic7')
    project = os.getenv('ROBOFLOW_PROJECT', 'main-fddlv')  # Project ID: main-fddlv
    model_version = int(os.getenv('ROBOFLOW_MODEL_VERSION', '2'))  # Using version 2
    
    print("=" * 70)
    print("🧪 Testing Currency Detection")
    print("=" * 70)
    print(f"Image: {image_path}")
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
                    print(f"✓ Connected! Using model version {v}")
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
    
    # Load image
    print(f"\n📷 Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        sys.exit(1)
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Convert BGR to RGB for Roboflow
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run detection
    print(f"\n🔍 Running detection...")
    try:
        # Higher confidence threshold to reduce false positives
        result = model.predict(image_rgb, confidence=75, overlap=30)
        predictions = result.json()
        detections = predictions.get('predictions', [])
        
        # Filter to only show 100 and 20 som
        all_detections = detections
        filtered_detections = [
            det for det in detections 
            if det.get('class', '').strip() in ['100', '20']
        ]
        
        if len(all_detections) != len(filtered_detections):
            print(f"⚠️  Filtered out {len(all_detections) - len(filtered_detections)} detection(s) (not 100 or 20 som)")
        
        # Remove overlapping detections - keep only the highest confidence one
        if len(filtered_detections) > 1:
            # Calculate overlap between detections
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
                
                # Calculate intersection
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
            
            # Remove overlapping detections (IoU > 0.5), keep highest confidence
            final_detections = []
            for det in filtered_detections:
                is_overlapping = False
                for existing in final_detections:
                    overlap = calculate_overlap(det, existing)
                    if overlap > 0.5:  # More than 50% overlap
                        # Keep the one with higher confidence
                        if det['confidence'] > existing['confidence']:
                            final_detections.remove(existing)
                            final_detections.append(det)
                        is_overlapping = True
                        break
                if not is_overlapping:
                    final_detections.append(det)
            
            if len(final_detections) < len(filtered_detections):
                print(f"⚠️  Removed {len(filtered_detections) - len(final_detections)} overlapping detection(s), kept highest confidence")
            
            detections = final_detections
        else:
            detections = filtered_detections
        
        print(f"\n{'=' * 70}")
        print(f"📊 Detection Results (100 and 20 som only)")
        print(f"{'=' * 70}")
        
        if not detections:
            print("❌ No 100 or 20 som currency detected in the image")
        else:
            print(f"✅ Found {len(detections)} detection(s):\n")
            
            for i, det in enumerate(detections, 1):
                roboflow_class = det.get('class', 'Unknown')
                class_name = map_class_name(roboflow_class)  # Apply swap mapping
                confidence = det.get('confidence', 0.0)
                
                print(f"  Detection {i}: {class_name} som")
                print(f"    Confidence: {confidence:.1%}")
                print()
        
        # Draw detections on image
        output_image = image.copy()
        # Color mapping - mapped to display names (after swap)
        colors = {
            '20': (0, 0, 255),      # Red for 20 som (display)
            '100': (0, 255, 0),     # Green for 100 som (display)
        }
        
        for det in detections:
            x = int(det['x'] - det['width'] / 2)
            y = int(det['y'] - det['height'] / 2)
            w = int(det['width'])
            h = int(det['height'])
            
            roboflow_class = det.get('class', 'Unknown')
            class_name = map_class_name(roboflow_class)  # Apply swap mapping
            confidence = det.get('confidence', 0.0)
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
            
            # Draw label
            label = f"{class_name} som ({confidence:.2%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            cv2.putText(output_image, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save output image as JPG
        output_path = image_path.rsplit('.', 1)[0] + '_result.jpg'
        cv2.imwrite(output_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"💾 Result saved: {output_path}")
        print(f"{'=' * 70}")
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function."""
    # Default to 100.jpg if no argument provided
    if len(sys.argv) < 2:
        image_path = "100.jpg"
        print(f"ℹ️  No image path provided, using default: {image_path}\n")
    else:
        image_path = sys.argv[1]
    
    test_image(image_path)


if __name__ == "__main__":
    main()

