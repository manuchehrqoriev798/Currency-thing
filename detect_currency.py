"""
Currency Detection Script - Uses trained model for detection
Run this after training with train_model.py
"""

import cv2
import numpy as np
import os
import joblib
import pickle
import time

class TrainedCurrencyDetector:
    def __init__(self, model_path="currency_model.pkl", scaler_path="currency_scaler.pkl", 
                 mappings_path="currency_mappings.pkl", currencies_folder="currencies"):
        """Initialize detector with trained model"""
        self.currencies_folder = currencies_folder
        
        # Load model
        print("Loading trained model...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Please run train_model.py first!")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}. Please run train_model.py first!")
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Mappings not found: {mappings_path}. Please run train_model.py first!")
        
        self.classifier = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        self.denom_to_label = mappings['denom_to_label']
        self.label_to_denom = mappings['label_to_denom']
        self.denominations = mappings['denominations']
        
        print("  ✓ Model loaded successfully")
        
        # Load templates for region detection
        self.templates = {}
        self.load_templates()
        self.detection_history = []
    
    def load_templates(self):
        """Load templates for detecting currency regions"""
        print("\nLoading templates for region detection...")
        image_extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        
        for denom in self.denominations.keys():
            denom_folder = os.path.join(self.currencies_folder, denom)
            self.templates[denom] = []
            
            if os.path.exists(denom_folder) and os.path.isdir(denom_folder):
                files = os.listdir(denom_folder)
                for filename in files:
                    if any(filename.lower().endswith(ext) for ext in image_extensions):
                        template = cv2.imread(os.path.join(denom_folder, filename), cv2.IMREAD_GRAYSCALE)
                        if template is not None:
                            self.templates[denom].append(template)
        
        print(f"  ✓ Loaded templates from {len([d for d in self.templates.values() if d])} denominations")
    
    def extract_features(self, image):
        """Extract features from an image (same as training)"""
        features = []
        
        # Resize image to standard size
        image_resized = cv2.resize(image, (200, 100))
        
        # 1. Color histogram features
        if len(image_resized.shape) == 3:
            for i in range(3):
                hist = cv2.calcHist([image_resized], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
        else:
            hist = cv2.calcHist([image_resized], [0], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # 2. Texture features
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) if len(image_resized.shape) == 3 else image_resized
        
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.mean(edges))
        features.append(np.std(edges))
        
        # 3. Statistical features
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.min(gray))
        features.append(np.max(gray))
        
        # 4. Shape features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            features.extend([area / 10000, circularity])
        else:
            features.extend([0, 0])
        
        # 5. HOG features
        hog_features = self.compute_hog_simple(gray)
        features.extend(hog_features)
        
        return np.array(features)
    
    def compute_hog_simple(self, gray):
        """Compute simplified HOG features"""
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 360
        
        hist, _ = np.histogram(angle, bins=8, range=(0, 360), weights=magnitude)
        return hist.tolist()
    
    def detect_currency_regions(self, frame):
        """Detect potential currency regions in frame"""
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.equalizeHist(processed)
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
        
        regions = []
        
        for denom, template_list in self.templates.items():
            for template in template_list:
                # Multi-scale template matching
                scales = [0.5, 0.75, 1.0, 1.25, 1.5]
                for scale in scales:
                    h, w = template.shape[:2]
                    scaled_template = cv2.resize(template, (int(w * scale), int(h * scale)))
                    
                    if scaled_template.shape[0] > processed.shape[0] or scaled_template.shape[1] > processed.shape[1]:
                        continue
                    
                    result = cv2.matchTemplate(processed, scaled_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= 0.6)
                    
                    for pt in zip(*locations[::-1]):
                        regions.append({
                            'bbox': (pt[0], pt[1], int(w * scale), int(h * scale)),
                            'scale': scale
                        })
        
        return regions
    
    def classify_regions(self, frame, regions):
        """Classify detected regions using trained model"""
        detected = []
        
        for region in regions:
            x, y, w, h = region['bbox']
            
            # Extract region of interest
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                continue
            
            try:
                # Extract features
                features = self.extract_features(roi)
                features = features.reshape(1, -1)
                
                # Normalize
                features_scaled = self.scaler.transform(features)
                
                # Predict
                label = self.classifier.predict(features_scaled)[0]
                confidence_scores = self.classifier.predict_proba(features_scaled)[0]
                confidence = max(confidence_scores)
                
                denom = self.label_to_denom[label]
                
                if confidence > 0.5:  # Minimum confidence threshold
                    detected.append({
                        'denomination': denom,
                        'value': self.denominations[denom],
                        'bbox': (x, y, w, h),
                        'confidence': confidence
                    })
            except Exception as e:
                continue
        
        return detected
    
    def detect_currencies(self, frame):
        """Main detection function"""
        # Detect regions
        regions = self.detect_currency_regions(frame)
        
        # Remove overlapping regions
        if len(regions) > 0:
            # Simple non-max suppression
            filtered_regions = []
            for i, r1 in enumerate(regions):
                x1, y1, w1, h1 = r1['bbox']
                overlap = False
                
                for r2 in filtered_regions:
                    x2, y2, w2, h2 = r2['bbox']
                    
                    # Check overlap
                    if not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
                        overlap = True
                        break
                
                if not overlap:
                    filtered_regions.append(r1)
            
            regions = filtered_regions
        
        # Classify regions
        detected = self.classify_regions(frame, regions)
        
        return detected
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        total_sum = 0
        colors = {
            '20': (0, 255, 0),    # Green
            '50': (255, 0, 0),    # Blue
            '100': (0, 165, 255), # Orange
            '500': (255, 0, 255), # Magenta
            '1000': (0, 255, 255),# Cyan
            '5000': (255, 255, 0) # Yellow
        }
        
        for det in detections:
            x, y, w, h = det['bbox']
            denom = det['denomination']
            value = det['value']
            conf = det['confidence']
            
            # Draw bounding box
            color = colors.get(denom, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw label
            label = f"{denom} som ({conf:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 5, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            total_sum += value
        
        # Draw total sum
        sum_text = f"Total: {total_sum} som"
        cv2.putText(frame, sum_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Draw count
        count_text = f"Items: {len(detections)}"
        cv2.putText(frame, count_text, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return frame, total_sum


def main():
    """Main function to run currency detection from camera"""
    print("=" * 60)
    print("Kyrgyz Som Currency Detection System")
    print("Using Trained Model")
    print("=" * 60)
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press SPACE to pause/resume detection")
    print("=" * 60 + "\n")
    
    # Initialize detector
    try:
        detector = TrainedCurrencyDetector()
        print("\n✅ Detector initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        print("\n⚠️  Make sure you have run train_model.py first!")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        print("Make sure your camera is connected and not being used by another application.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Camera initialized. Starting detection...\n")
    
    paused = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            if not paused:
                # Detect currencies
                detections = detector.detect_currencies(frame)
                
                # Draw detections
                display_frame, total_sum = detector.draw_detections(display_frame, detections)
                
                # Update detection history
                detector.detection_history.append({
                    'timestamp': time.time(),
                    'detections': detections,
                    'total': total_sum
                })
            
            # Add pause indicator
            if paused:
                cv2.putText(display_frame, "PAUSED - Press SPACE to resume", 
                           (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Currency Detection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detected_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Frame saved as {filename}")
            elif key == ord(' '):
                paused = not paused
                print("Detection paused" if paused else "Detection resumed")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera released. Goodbye!")
        
        # Print summary
        if detector.detection_history:
            latest = detector.detection_history[-1]
            print(f"\nFinal Detection Summary:")
            print(f"  Total items detected: {len(latest['detections'])}")
            print(f"  Total sum: {latest['total']} som")


if __name__ == "__main__":
    main()

