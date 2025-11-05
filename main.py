#!/usr/bin/env python3
"""
Kyrgyz Som Currency Detection - Main Script

This script:
1. Removes old trained model files (.pkl)
2. Trains a new model from images in currencies/ folder
3. Runs real-time currency detection using camera
"""

import os
# Set Qt backend before importing cv2 to avoid plugin issues
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import glob

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Currency denominations
DENOMINATIONS = ['20', '50', '100', '500', '1000', '5000']

# Feature extraction parameters
TARGET_SIZE = (200, 100)  # Width, Height
HIST_BINS = 32
HOG_BINS = 8

# Detection parameters
TEMPLATE_MATCH_THRESHOLD = 0.6
MIN_DETECTION_AREA = 5000
MAX_DETECTION_AREA = 500000
NMS_THRESHOLD = 0.3


def remove_old_models():
    """Remove old trained model files."""
    model_files = ['currency_model.pkl', 'currency_scaler.pkl', 'currency_mappings.pkl']
    removed = []
    for model_file in model_files:
        if os.path.exists(model_file):
            os.remove(model_file)
            removed.append(model_file)
            print(f"  ✓ Removed {model_file}")
    if removed:
        print(f"\nRemoved {len(removed)} old model file(s)\n")
    else:
        print("No old model files to remove\n")


def load_images_from_folder(folder_path):
    """Load all images from a folder."""
    images = []
    if not os.path.exists(folder_path):
        return images
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = Path(filename).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                img = cv2.imread(file_path)
                if img is not None:
                    images.append(img)
    
    return images


def extract_features(image):
    """Extract 112 features from an image."""
    # Resize to consistent size
    resized = cv2.resize(image, TARGET_SIZE)
    
    features = []
    
    # 1. Color histogram features (96 features: 32 bins × 3 channels)
    b, g, r = cv2.split(resized)
    hist_b = cv2.calcHist([b], [0], None, [HIST_BINS], [0, 256]).flatten()
    hist_g = cv2.calcHist([g], [0], None, [HIST_BINS], [0, 256]).flatten()
    hist_r = cv2.calcHist([r], [0], None, [HIST_BINS], [0, 256]).flatten()
    features.extend(hist_b)
    features.extend(hist_g)
    features.extend(hist_r)
    
    # 2. Edge features (2 features: mean and std of Canny edges)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.mean(edges))
    features.append(np.std(edges))
    
    # 3. Statistical features (4 features: mean, std, min, max of grayscale)
    features.append(np.mean(gray))
    features.append(np.std(gray))
    features.append(np.min(gray))
    features.append(np.max(gray))
    
    # 4. Shape features (2 features: area, circularity)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        features.append(area / 10000.0)  # Normalized area
        features.append(circularity)
    else:
        features.append(0.0)
        features.append(0.0)
    
    # 5. HOG-like gradient orientation histogram (8 features)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 360
    
    hist, _ = np.histogram(angle, bins=HOG_BINS, range=(0, 360), weights=magnitude)
    hist = hist / (np.sum(hist) + 1e-6)  # Normalize
    features.extend(hist)
    
    return np.array(features, dtype=np.float32)


def augment_data(images, labels):
    """Augment data by horizontal flipping."""
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        augmented_images.append(img)
        augmented_labels.append(label)
        flipped = cv2.flip(img, 1)
        augmented_images.append(flipped)
        augmented_labels.append(label)
    
    return augmented_images, augmented_labels


def train_model():
    """Train the currency detection model."""
    print("=" * 70)
    print("TRAINING CURRENCY DETECTION MODEL")
    print("=" * 70)
    print()
    
    currencies_dir = "currencies"
    if not os.path.exists(currencies_dir):
        print("❌ Error: 'currencies' folder not found!")
        print("   Please create the folder structure with currency images.")
        return False
    
    print("Loading currency templates from folders...")
    print()
    
    all_images = []
    all_labels = []
    
    for denom in DENOMINATIONS:
        folder_path = os.path.join(currencies_dir, denom)
        images = load_images_from_folder(folder_path)
        
        if images:
            all_images.extend(images)
            all_labels.extend([denom] * len(images))
            print(f"  ✓ Loaded {len(images)} template(s) for {denom} som")
        else:
            print(f"  ⚠ No images found in {folder_path}")
    
    print()
    
    if len(all_images) == 0:
        print("❌ Error: No training images found!")
        print("   Please add images to the currencies/ folders.")
        return False
    
    print(f"Total images loaded: {len(all_images)}")
    print()
    
    # Data augmentation
    print("Applying data augmentation (horizontal flip)...")
    augmented_images, augmented_labels = augment_data(all_images, all_labels)
    print(f"After augmentation: {len(augmented_images)} images")
    print()
    
    # Extract features
    print("Extracting features from images...")
    features_list = []
    for i, img in enumerate(augmented_images):
        features = extract_features(img)
        features_list.append(features)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(augmented_images)} images...")
    
    X = np.array(features_list)
    y = np.array(augmented_labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print()
    
    # Split into training and test sets
    # Check if we have enough samples for stratified splitting
    unique_labels = len(set(y))
    test_size = 0.2
    
    # For stratified split, test set must have at least as many samples as classes
    # With test_size=0.2, we need at least num_classes / test_size total samples
    min_samples_for_stratified = int(np.ceil(unique_labels / test_size))
    
    # Also check that each class has at least 2 samples
    min_samples_per_class = 2
    each_class_has_enough = all(np.sum(y == label) >= min_samples_per_class for label in set(y))
    
    # Check if we can do stratified split
    can_stratify = len(X) >= min_samples_for_stratified and each_class_has_enough
    
    if can_stratify:
        print("Splitting data into training and test sets (stratified)...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            has_test_set = True
        except ValueError:
            # Fallback if stratified split still fails
            print("Stratified split failed, using all data for training...")
            X_train, X_test, y_train, y_test = X, np.array([]), y, np.array([])
            has_test_set = False
    else:
        # For small datasets, use all data for training
        print("Dataset too small for train/test split. Using all data for training...")
        X_train, X_test, y_train, y_test = X, np.array([]), y, np.array([])
        has_test_set = False
    
    print(f"Training samples: {len(X_train)}")
    if has_test_set:
        print(f"Test samples: {len(X_test)}")
    print()
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if has_test_set:
        X_test_scaled = scaler.transform(X_test)
    print()
    
    # Train Random Forest classifier
    print("Training Random Forest classifier (100 trees)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    print()
    
    # Evaluate
    print("Evaluating model...")
    train_score = clf.score(X_train_scaled, y_train)
    print(f"Training Accuracy: {train_score * 100:.2f}%")
    if has_test_set:
        test_score = clf.score(X_test_scaled, y_test)
        print(f"Test Accuracy: {test_score * 100:.2f}%")
    else:
        print("(Test set skipped - dataset too small)")
    print()
    
    # Create label mappings
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(y)))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Save model files
    print("Saving model files...")
    try:
        joblib.dump(clf, 'currency_model.pkl')
        joblib.dump(scaler, 'currency_scaler.pkl')
        joblib.dump({'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label}, 
                   'currency_mappings.pkl')
        print("  ✓ currency_model.pkl")
        print("  ✓ currency_scaler.pkl")
        print("  ✓ currency_mappings.pkl")
        print()
        print("✓ Model saved successfully!")
        print()
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False
    
    print("=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print()
    return True


def find_camera():
    """Find an available camera."""
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return i
            cap.release()
    return None


def load_templates():
    """Load currency templates for template matching."""
    templates = {}
    currencies_dir = "currencies"
    
    for denom in DENOMINATIONS:
        folder_path = os.path.join(currencies_dir, denom)
        images = load_images_from_folder(folder_path)
        if images:
            # Use the first image as template
            templates[denom] = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    
    return templates


def non_max_suppression(boxes, scores, overlap_threshold):
    """Apply non-maximum suppression to remove overlapping detections."""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Convert boxes to (x1, y1, x2, y2) format
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[1:]]
        
        inds = np.where(overlap <= overlap_threshold)[0]
        order = order[inds + 1]
    
    return keep


def detect_currencies(frame, model, scaler, mappings, templates):
    """Detect currencies using contour-based detection (faster than template matching)."""
    detections = []
    
    # Resize frame for faster processing (keep aspect ratio)
    scale_factor = 0.5
    h, w = frame.shape[:2]
    small_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
    small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(small_gray, (5, 5), 0)
    
    # Use adaptive threshold to find currency-like regions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        # Scale area back to original frame size
        scaled_area = area / (scale_factor * scale_factor)
        
        # Filter by area
        if MIN_DETECTION_AREA <= scaled_area <= MAX_DETECTION_AREA:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Scale coordinates back to original frame
            x = int(x / scale_factor)
            y = int(y / scale_factor)
            w = int(w / scale_factor)
            h = int(h / scale_factor)
            
            # Make sure we're within frame bounds
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w < 50 or h < 50:  # Skip too small regions
                continue
            
            # Extract region of interest
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                continue
            
            # Extract features and classify
            try:
                features = extract_features(roi)
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                confidence = model.predict_proba(features_scaled)[0].max()
                
                # Only keep high-confidence detections
                if confidence > 0.6:
                    detections.append({
                        'box': (x, y, w, h),
                        'denomination': prediction,
                        'confidence': confidence
                    })
            except Exception:
                continue
    
    # Apply non-maximum suppression
    if detections:
        boxes = [d['box'] for d in detections]
        scores = [d['confidence'] for d in detections]
        keep_indices = non_max_suppression(boxes, scores, NMS_THRESHOLD)
        detections = [detections[i] for i in keep_indices]
    
    return detections


def run_detection():
    """Run real-time currency detection."""
    print("=" * 70)
    print("STARTING CURRENCY DETECTION")
    print("=" * 70)
    print()
    
    # Load model files
    if not os.path.exists('currency_model.pkl'):
        print("❌ Error: Model files not found!")
        print("   Please train the model first.")
        return
    
    print("Loading trained model...")
    try:
        model = joblib.load('currency_model.pkl')
        scaler = joblib.load('currency_scaler.pkl')
        mappings = joblib.load('currency_mappings.pkl')
        print("✓ Model loaded successfully")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load templates
    templates = load_templates()
    if not templates:
        print("❌ Error: No templates found in currencies/ folder!")
        return
    
    print(f"Loaded {len(templates)} template(s)")
    print()
    
    # Find camera
    print("Searching for camera...")
    camera_idx = find_camera()
    if camera_idx is None:
        print("❌ Error: No camera found!")
        return
    
    print(f"✓ Camera found at index {camera_idx}")
    print()
    
    # Open camera
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        return
    
    # Set camera properties (lower resolution for better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera opened successfully")
    print()
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  SPACE - Pause/Resume")
    print()
    print("Starting detection...")
    print()
    
    paused = False
    frame_count = 0
    detections = []  # Store detections between frames
    
    # Color mapping for denominations
    colors = {
        '20': (0, 255, 0),    # Green
        '50': (255, 0, 0),    # Blue
        '100': (0, 0, 255),   # Red
        '500': (255, 255, 0), # Cyan
        '1000': (255, 0, 255), # Magenta
        '5000': (0, 255, 255)  # Yellow
    }
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                frame_count += 1
                
                # Detect currencies every 3 frames for better responsiveness
                # (contour-based detection is faster than template matching)
                if frame_count % 3 == 0:
                    detections = detect_currencies(frame, model, scaler, mappings, templates)
                # Keep previous detections for smoother display between detection frames
            
            # Draw detections on frame
            display_frame = frame.copy()
            total_sum = 0
            item_count = 0
            
            for det in detections:
                x, y, w, h = det['box']
                denom = det['denomination']
                conf = det['confidence']
                
                # Draw bounding box
                color = colors.get(denom, (255, 255, 255))
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{denom} som ({conf:.2f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                cv2.putText(display_frame, label, (x, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                total_sum += int(denom)
                item_count += 1
            
            # Draw total sum and count
            info_text = f"Total: {total_sum} som | Items: {item_count}"
            cv2.rectangle(display_frame, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(display_frame, info_text, (15, 35),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if paused:
                pause_text = "PAUSED"
                cv2.putText(display_frame, pause_text, (15, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Currency Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Saved frame to {filename}")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
    
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


def main():
    """Main function."""
    print("=" * 70)
    print("KYRGYZ SOM CURRENCY DETECTION SYSTEM")
    print("=" * 70)
    print()
    
    # Step 1: Remove old models
    print("Step 1: Removing old trained models...")
    remove_old_models()
    
    # Step 2: Train model
    print("Step 2: Training model from currencies/ folder...")
    if not train_model():
        print("Training failed. Exiting.")
        return
    
    # Step 3: Run detection
    print("Step 3: Starting currency detection...")
    run_detection()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

