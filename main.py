#!/usr/bin/env python3
"""
Kyrgyz Som Currency Detection - Main Script

This script:
1. Removes old trained model files (.pkl)
2. Trains a new deep learning model from images in currencies/ folder
3. Runs real-time currency detection using camera
"""

import os
import sys
import warnings
# Set Qt backend before importing cv2 to avoid plugin issues
os.environ['QT_QPA_PLATFORM'] = 'xcb'
# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import cv2
import threading
import time
# Set OpenCV to only show errors (suppress warnings) - if available
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    # Older OpenCV versions don't have setLogLevel
    pass
# Suppress Python warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
import glob
import joblib

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Only JPG/JPEG formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg'}


class SuppressStderr:
    """Context manager to suppress stderr output."""
    def __init__(self):
        self.original_stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
    
    def __enter__(self):
        sys.stderr = self.devnull
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr
        self.devnull.close()
        return False

# Currency denominations
DENOMINATIONS = ['20', '50', '100', '200', '500', '1000', '5000']

# Model parameters
IMG_SIZE = (224, 224)  # Standard size for transfer learning
BATCH_SIZE = 32
EPOCHS = 50  # Heavy training with many epochs
LEARNING_RATE = 0.0001

# Detection parameters
MIN_CONFIDENCE = 0.3  # Lower threshold for better detection
MIN_DETECTION_AREA = 5000
MAX_DETECTION_AREA = 500000
NMS_THRESHOLD = 0.3


def remove_old_models():
    """Remove old trained model files - ALWAYS delete before training."""
    model_files = ['currency_model.pkl', 'currency_scaler.pkl', 'currency_mappings.pkl', 'currency_model.h5']
    removed = []
    for model_file in model_files:
        if os.path.exists(model_file):
            os.remove(model_file)
            removed.append(model_file)
            print(f"  ✓ Removed {model_file}")
    if removed:
        print(f"\nRemoved {len(removed)} old model file(s)")
        print("  (Old models deleted - will train fresh model)\n")
    else:
        print("No old model files to remove")
        print("  (Will train new model)\n")


def load_images_from_folder(folder_path):
    """Load only JPG/JPEG images from a folder."""
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


def create_heavy_augmentation():
    """Create heavy data augmentation generator."""
    return ImageDataGenerator(
        rotation_range=30,           # Rotate up to 30 degrees
        width_shift_range=0.2,       # Shift horizontally
        height_shift_range=0.2,      # Shift vertically
        shear_range=0.2,             # Shear transformation
        zoom_range=0.3,               # Zoom in/out
        horizontal_flip=True,         # Flip horizontally
        vertical_flip=False,          # Don't flip vertically (currency orientation matters)
        brightness_range=[0.7, 1.3], # Brightness variation
        channel_shift_range=50,       # Color channel shifts
        fill_mode='nearest',
        rescale=1./255
    )


def build_model(num_classes):
    """Build a deep learning model using transfer learning."""
    # Use EfficientNetB0 as base (more powerful than MobileNetV2)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with learning rate scheduling
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model, base_model


def prepare_data():
    """Load and prepare all training data."""
    print("Loading currency images from folders...")
    print()
    
    all_images = []
    all_labels = []
    
    for denom in DENOMINATIONS:
        folder_path = os.path.join("currencies", denom)
        images = load_images_from_folder(folder_path)
        
        if images:
            all_images.extend(images)
            all_labels.extend([denom] * len(images))
            print(f"  ✓ Loaded {len(images)} image(s) for {denom} som")
        else:
            print(f"  ⚠ No images found in {folder_path}")
    
    print()
    
    if len(all_images) == 0:
        print("❌ Error: No training images found!")
        print("   Please add JPG images to the currencies/ folders.")
        return None, None
    
    print(f"Total images loaded: {len(all_images)}")
    print()
    
    # Resize all images to model input size
    print("Resizing images to {}...".format(IMG_SIZE))
    resized_images = []
    for img in all_images:
        resized = cv2.resize(img, IMG_SIZE)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        resized_images.append(resized)
    
    X = np.array(resized_images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    y = np.array(all_labels)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {set(y)}")
    print()
    
    return X, y


def train_model():
    """Train the currency detection model with heavy training."""
    print("=" * 70)
    print("TRAINING CURRENCY DETECTION MODEL (DEEP LEARNING)")
    print("=" * 70)
    print()
    
    # Prepare data
    X, y = prepare_data()
    if X is None:
        return False
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    print()
    
    # Split data
    print("Splitting data into training and validation sets...")
    if len(X) >= num_classes * 2:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        has_val = True
    else:
        print("Dataset too small for validation split. Using all data for training...")
        X_train, X_val, y_train, y_val = X, X, y_encoded, y_encoded
        has_val = False
    
    print(f"Training samples: {len(X_train)}")
    if has_val:
        print(f"Validation samples: {len(X_val)}")
    print()
    
    # Build model
    print("Building deep learning model with transfer learning (EfficientNetB0)...")
    model, base_model = build_model(num_classes)
    print(f"Total parameters: {model.count_params():,}")
    print()
    
    # Create callbacks for heavy training (Stage 1)
    callbacks_stage1 = [
        callbacks.EarlyStopping(
            monitor='val_loss' if has_val else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss' if has_val else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_model_stage1.weights.h5',
            monitor='val_accuracy' if has_val else 'accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    
    # Create callbacks for fine-tuning (Stage 2)
    callbacks_stage2 = [
        callbacks.EarlyStopping(
            monitor='val_loss' if has_val else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss' if has_val else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_model_stage2.weights.h5',
            monitor='val_accuracy' if has_val else 'accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    
    # Create heavy augmentation
    print("Creating heavy data augmentation generator...")
    datagen = create_heavy_augmentation()
    
    # Stage 1: Train with frozen base model
    print("=" * 70)
    print("STAGE 1: Training with frozen base model")
    print("=" * 70)
    print()
    
    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS // 2,
        validation_data=(X_val, y_val) if has_val else None,
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    # Stage 2: Fine-tune with unfrozen base model
    print()
    print("=" * 70)
    print("STAGE 2: Fine-tuning with unfrozen base model")
    print("=" * 70)
    print()
    
    # Unfreeze base model layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze early layers
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    history2 = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS // 2,
        validation_data=(X_val, y_val) if has_val else None,
        callbacks=callbacks_stage2,
        verbose=1
    )
    
    # Load best model from stage 2 (final fine-tuned model)
    if os.path.exists('best_model_stage2.weights.h5'):
        print("\nLoading best model weights from stage 2...")
        model.load_weights('best_model_stage2.weights.h5')
    
    print()
    print("=" * 70)
    print("EVALUATING MODEL PERFORMANCE")
    print("=" * 70)
    print()
    
    # Evaluate on training set
    train_loss, train_acc, train_top3 = model.evaluate(X_train, y_train, verbose=0)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Training Top-3 Accuracy: {train_top3 * 100:.2f}%")
    print()
    
    # Evaluate on validation set
    if has_val:
        val_loss, val_acc, val_top3 = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")
        print(f"Validation Top-3 Accuracy: {val_top3 * 100:.2f}%")
        print()
        
        # Per-class accuracy
        y_val_pred = model.predict(X_val, verbose=0)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        
        print("Per-Class Validation Accuracy:")
        for i, class_name in enumerate(label_encoder.classes_):
            mask = y_val == i
            if np.sum(mask) > 0:
                class_acc = np.mean(y_val_pred_classes[mask] == y_val[mask]) * 100
                count = np.sum(mask)
                print(f"  {class_name} som: {class_acc:.2f}% ({count} samples)")
        print()
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_val, y_val_pred_classes, 
                                   target_names=label_encoder.classes_))
        print()
    
    # Save model
    print("=" * 70)
    print("Saving model files...")
    print("=" * 70)
    print()
    
    # Save Keras model
    model.save('currency_model.h5')
    print("  ✓ currency_model.h5")
    
    # Save label encoder and mappings
    mappings = {
        'label_encoder': label_encoder,
        'label_to_idx': {label: idx for idx, label in enumerate(label_encoder.classes_)},
        'idx_to_label': {idx: label for idx, label in enumerate(label_encoder.classes_)}
    }
    joblib.dump(mappings, 'currency_mappings.pkl')
    print("  ✓ currency_mappings.pkl")
    
    # Also save as pickle for compatibility (though we'll use .h5)
    joblib.dump(model, 'currency_model.pkl')
    joblib.dump(None, 'currency_scaler.pkl')  # Dummy scaler for compatibility
    print("  ✓ currency_model.pkl (compatibility)")
    print("  ✓ currency_scaler.pkl (compatibility)")
    print()
    
    # Clean up temporary checkpoint files
    for checkpoint_file in ['best_model_stage1.weights.h5', 'best_model_stage2.weights.h5']:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
    
    print("=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print()
    
    return True


def find_camera(preferred_index=None):
    """Find an available camera, optionally starting from a preferred index."""
    # If preferred index is specified, try it first
    if preferred_index is not None:
        try:
            # Suppress stderr to avoid OpenCV warnings
            with SuppressStderr():
                cap = cv2.VideoCapture(preferred_index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.release()
                        return preferred_index
                    cap.release()
        except Exception:
            pass
    
    # Otherwise, search all available cameras (silently)
    available_cameras = []
    print("  Scanning for cameras...")
    for i in range(20):  # Increased range to find phone cameras
        try:
            # Suppress stderr to avoid OpenCV warnings
            with SuppressStderr():
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                        print(f"    ✓ Camera found at index {i}")
                    cap.release()
        except Exception:
            # Silently skip cameras that can't be opened
            pass
    
    if available_cameras:
        # Return the first available camera (or last one if multiple, as phone is usually added last)
        return available_cameras[-1] if len(available_cameras) > 1 else available_cameras[0]
    
    return None


def list_cameras():
    """List all available cameras."""
    cameras = []
    print("\nScanning for cameras...")
    for i in range(20):
        try:
            # Suppress stderr to avoid OpenCV warnings
            with SuppressStderr():
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Try to get camera name/info
                        backend = cap.getBackendName()
                        cameras.append({
                            'index': i,
                            'backend': backend,
                            'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                        })
                        print(f"  Camera {i}: {backend} - Resolution: {cameras[-1]['resolution']}")
                    cap.release()
        except Exception:
            # Silently skip cameras that can't be opened
            pass
    return cameras


def non_max_suppression(boxes, scores, overlap_threshold):
    """Apply non-maximum suppression to remove overlapping detections."""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
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


def detect_currencies(frame, model, mappings):
    """Detect currencies using sliding window approach with batch processing."""
    detections = []
    h, w = frame.shape[:2]
    
    # First, try the entire frame (fastest and most accurate if currency fills frame)
    try:
        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        prediction = model.predict(frame_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx]
        
        if confidence >= MIN_CONFIDENCE:
            predicted_label = mappings['idx_to_label'][predicted_class_idx]
            detections.append({
                'box': (0, 0, w, h),
                'denomination': predicted_label,
                'confidence': confidence,
                'distance': "Full Frame",
                'center': (w // 2, h // 2)
            })
            # If full frame detection is confident enough, return it
            if confidence >= 0.7:
                return detections
    except Exception:
        pass
    
    # Define window sizes (adjusted for better coverage)
    window_sizes = [
        (int(w * 0.6), int(h * 0.3)),   # Large window
        (int(w * 0.4), int(h * 0.2)),   # Medium window
        (int(w * 0.3), int(h * 0.15)),  # Small window
    ]
    
    # Ensure minimum window sizes
    window_sizes = [(max(w_size, 100), max(h_size, 50)) for w_size, h_size in window_sizes]
    
    step_size = 60  # Smaller step for better coverage
    
    # Collect all ROIs and their metadata for batch processing
    rois_data = []
    
    # Process each window size
    for win_w, win_h in window_sizes:
        for y in range(0, h - win_h + 1, step_size):
            for x in range(0, w - win_w + 1, step_size):
                roi = frame[y:y+win_h, x:x+win_w]
                
                if roi.size == 0:
                    continue
                
                try:
                    # Preprocess ROI
                    roi_resized = cv2.resize(roi, IMG_SIZE)
                    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                    roi_normalized = roi_rgb.astype(np.float32) / 255.0
                    
                    rois_data.append({
                        'roi': roi_normalized,
                        'box': (x, y, win_w, win_h),
                        'win_w': win_w
                    })
                except Exception:
                    continue
    
    # Batch process all ROIs at once (much faster than individual predictions)
    if rois_data:
        try:
            batch = np.array([data['roi'] for data in rois_data])
            predictions_batch = model.predict(batch, verbose=0, batch_size=32)
            
            # Track max confidence for debugging
            max_conf = 0.0
            
            # Process predictions
            for i, pred in enumerate(predictions_batch):
                predicted_class_idx = np.argmax(pred)
                confidence = pred[predicted_class_idx]
                max_conf = max(max_conf, confidence)
                
                if confidence >= MIN_CONFIDENCE:
                    predicted_label = mappings['idx_to_label'][predicted_class_idx]
                    data = rois_data[i]
                    x, y, win_w, win_h = data['box']
                    
                    # Estimate distance based on window size relative to frame
                    win_ratio = data['win_w'] / w
                    if win_ratio >= 0.5:
                        distance_category = "Close"
                    elif win_ratio >= 0.3:
                        distance_category = "Medium"
                    else:
                        distance_category = "Far"
                    
                    detections.append({
                        'box': (x, y, win_w, win_h),
                        'denomination': predicted_label,
                        'confidence': confidence,
                        'distance': distance_category,
                        'center': (x + win_w // 2, y + win_h // 2)
                    })
            
            # Debug: Print max confidence if no detections (only occasionally to avoid spam)
            if len(detections) == 0 and len(rois_data) > 0:
                import random
                if random.random() < 0.1:  # 10% chance to print
                    print(f"Debug: No detections. Max confidence: {max_conf:.3f}, Threshold: {MIN_CONFIDENCE}, ROIs checked: {len(rois_data)}")
        except Exception as e:
            print(f"Batch prediction error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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
    
    # Load model
    model_path = 'currency_model.h5'
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found!")
        print("   Please train the model first.")
        return
    
    print("Loading trained model...")
    try:
        model = keras.models.load_model(model_path)
        mappings = joblib.load('currency_mappings.pkl')
        print("✓ Model loaded successfully")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check for camera index from environment variable or command line
    import sys
    camera_idx = None
    
    # Check command line argument
    if len(sys.argv) > 1:
        try:
            camera_idx = int(sys.argv[1])
            print(f"Using camera index from command line: {camera_idx}")
        except ValueError:
            print(f"⚠ Invalid camera index: {sys.argv[1]}. Searching automatically...")
    
    # Check environment variable
    if camera_idx is None:
        env_camera = os.environ.get('CAMERA_INDEX')
        if env_camera:
            try:
                camera_idx = int(env_camera)
                print(f"Using camera index from environment: {camera_idx}")
            except ValueError:
                print(f"⚠ Invalid camera index in environment: {env_camera}")
    
    # Find camera
    print("Searching for camera...")
    if camera_idx is None:
        # List all cameras first (this will also scan, but we'll reuse the results)
        cameras = list_cameras()
        if len(cameras) > 1:
            print(f"\n  Found {len(cameras)} camera(s). Using the last one (usually your phone).")
            print(f"  To use a specific camera, run: python main.py <camera_index>")
            print(f"  Or set environment variable: export CAMERA_INDEX=<camera_index>")
            # Use the camera we already found from list_cameras
            camera_idx = cameras[-1]['index'] if len(cameras) > 1 else cameras[0]['index']
        elif len(cameras) == 1:
            camera_idx = cameras[0]['index']
        else:
            # If list_cameras didn't find any, try find_camera (but it should have found them)
            camera_idx = find_camera()
    
    if camera_idx is None:
        print("❌ Error: No camera found!")
        print("\nTroubleshooting:")
        print("  1. Make sure your phone is connected via USB")
        print("  2. Enable USB debugging/File transfer mode on your phone")
        print("  3. Install a webcam app on your phone (DroidCam, iVCam, etc.)")
        print("  4. Try: python main.py <camera_index> to specify camera manually")
        return
    
    print(f"✓ Camera found at index {camera_idx}")
    print()
    
    # Open camera with V4L2 backend if available
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        return
    
    # Tune camera properties for low-latency smooth preview
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    # Prefer MJPEG to reduce CPU usage
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    except Exception:
        pass
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
    detections = []
    latest_frame = None

    # Thread-safe latest frame buffer
    frame_lock = threading.Lock()
    stop_event = threading.Event()

    def grab_frames():
        nonlocal latest_frame, frame_count
        # Clear some initial frames to avoid startup lag
        for _ in range(10):
            cap.read()
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Avoid tight loop if camera hiccups
                time.sleep(0.005)
                continue
            with frame_lock:
                latest_frame = frame
                frame_count += 1

    # Background detection worker
    detection_lock = threading.Lock()
    detection_in_progress = False

    def detection_worker():
        nonlocal detections, detection_in_progress
        # Run periodically, picking the most recent frame
        while not stop_event.is_set():
            if paused:
                time.sleep(0.05)
                continue
            # Sample every ~8 frames (~4-8 fps inference depending on camera fps)
            with frame_lock:
                current_frame = None if latest_frame is None else latest_frame.copy()
                current_count = frame_count
            if current_frame is None:
                time.sleep(0.01)
                continue
            # Only infer on every 8th frame snapshot
            if current_count % 8 != 0:
                time.sleep(0.005)
                continue
            if detection_in_progress:
                time.sleep(0.001)
                continue
            detection_in_progress = True
            try:
                new_dets = detect_currencies(current_frame, model, mappings)
                with detection_lock:
                    detections = new_dets
            except Exception as e:
                print(f"Detection error: {e}")
            finally:
                detection_in_progress = False
            # Small sleep to avoid hot loop
            time.sleep(0.001)
    
    # Color mapping for denominations
    colors = {
        '20': (0, 255, 0),      # Green
        '50': (255, 0, 0),      # Blue
        '100': (0, 0, 255),     # Red
        '200': (255, 128, 0),   # Orange
        '500': (255, 255, 0),   # Cyan
        '1000': (255, 0, 255),  # Magenta
        '5000': (0, 255, 255)   # Yellow
    }
    
    # Start threads
    grabber_thread = threading.Thread(target=grab_frames, daemon=True)
    infer_thread = threading.Thread(target=detection_worker, daemon=True)
    grabber_thread.start()
    infer_thread.start()
    
    try:
        while True:
            # Pull the most recent frame for display
            with frame_lock:
                display_frame = None if latest_frame is None else latest_frame.copy()
            if display_frame is None:
                # No frame yet; wait briefly
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            detected_count = 0
            
            # Use latest cached detections from background worker
            with detection_lock:
                current_dets = list(detections)
            for det in current_dets:
                x, y, w, h = det['box']
                denom = det['denomination']
                conf = det['confidence']
                distance = det.get('distance', 'Unknown')
                center_x, center_y = det.get('center', (x + w // 2, y + h // 2))
                
                # Draw bounding box
                color = colors.get(denom, (255, 255, 255))
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw center point
                cv2.circle(display_frame, (center_x, center_y), 5, color, -1)
                
                # Draw label
                label = f"{denom} som ({conf:.2f})"
                distance_text = f"Dist: {distance}"
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                dist_size, _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                box_height = label_size[1] + dist_size[1] + 15
                box_width = max(label_size[0], dist_size[0]) + 10
                cv2.rectangle(display_frame, (x, y - box_height), 
                            (x + box_width, y), color, -1)
                
                y_offset = y - 5
                cv2.putText(display_frame, label, (x + 5, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                y_offset -= dist_size[1] + 5
                cv2.putText(display_frame, distance_text, (x + 5, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                detected_count += 1
            
            # Draw detection count and status
            info_text = f"Detected: {detected_count} currency note(s)"
            status_color = (0, 255, 0) if detected_count > 0 else (255, 255, 255)
            cv2.rectangle(display_frame, (10, 10), (380, 85), (0, 0, 0), -1)
            cv2.putText(display_frame, info_text, (15, 35),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show detection status
            if detection_in_progress:
                status_text = "Detecting..."
                cv2.putText(display_frame, status_text, (15, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                status_text = "Ready"
                cv2.putText(display_frame, status_text, (15, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if paused:
                pause_text = "PAUSED"
                cv2.putText(display_frame, pause_text, (15, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Currency Detection', display_frame)
            
            # Use waitKey with small delay to allow display to update
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
        stop_event.set()
        try:
            grabber_thread.join(timeout=0.5)
            infer_thread.join(timeout=0.5)
        except Exception:
            pass
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
    print("Step 2: Training deep learning model from currencies/ folder...")
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
