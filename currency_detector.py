"""
Kyrgyz Som Currency Detection System
Detects currencies from camera feed and calculates total sum
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time

class CurrencyDetector:
    def __init__(self, currencies_folder="currencies"):
        """Initialize the currency detector with templates from folders"""
        self.currencies_folder = currencies_folder
        self.denominations = {
            '20': 20,
            '50': 50,
            '100': 100,
            '500': 500,
            '1000': 1000,
            '5000': 5000
        }
        self.templates = {}  # Dictionary: {denomination: [list of template images]}
        self.load_templates()
        self.detection_history = []
        
    def load_templates(self):
        """Load all currency template images from folder structure"""
        print("Loading currency templates from folders...")
        total_templates = 0
        
        # Supported image extensions
        image_extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        
        for denom in self.denominations.keys():
            denom_folder = os.path.join(self.currencies_folder, denom)
            self.templates[denom] = []
            
            if not os.path.exists(denom_folder):
                print(f"  ✗ Folder not found: {denom_folder}")
                continue
            
            if not os.path.isdir(denom_folder):
                print(f"  ✗ Not a directory: {denom_folder}")
                continue
            
            # Load all images from the denomination folder
            files = os.listdir(denom_folder)
            loaded_count = 0
            
            for filename in files:
                file_path = os.path.join(denom_folder, filename)
                
                # Check if it's an image file
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    template = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    if template is not None:
                        # Convert to grayscale for matching
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        self.templates[denom].append(template_gray)
                        loaded_count += 1
                        total_templates += 1
            
            if loaded_count > 0:
                print(f"  ✓ Loaded {loaded_count} template(s) for {denom} som")
            else:
                print(f"  ✗ No templates found in {denom_folder}")
        
        if total_templates == 0:
            raise ValueError(f"No templates loaded! Please add images to folders in '{self.currencies_folder}/' directory.")
        
        print(f"\nTotal templates loaded: {total_templates} from {len([d for d in self.templates.values() if d])} denominations")
    
    def preprocess_image(self, image):
        """Enhance image for better detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for contrast enhancement
        equalized = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        return blurred
    
    def detect_currency_template_matching(self, image, template, threshold=0.7):
        """Detect currency using template matching"""
        # Resize template if image is too small or large
        h, w = template.shape[:2]
        img_h, img_w = image.shape[:2]
        
        # Multi-scale template matching
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        all_matches = []
        
        for scale in scales:
            scaled_template = cv2.resize(template, (int(w * scale), int(h * scale)))
            if scaled_template.shape[0] > img_h or scaled_template.shape[1] > img_w:
                continue
                
            result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                all_matches.append({
                    'point': pt,
                    'scale': scale,
                    'confidence': result[pt[1], pt[0]]
                })
        
        return all_matches
    
    def non_max_suppression(self, matches, template_height, template_width, overlap_threshold=0.3):
        """Remove overlapping detections"""
        if len(matches) == 0:
            return []
        
        # Convert to bounding boxes
        boxes = []
        confidences = []
        for match in matches:
            pt = match['point']
            scale = match['scale']
            w = int(template_width * scale)
            h = int(template_height * scale)
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
            confidences.append(match['confidence'])
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, overlap_threshold)
        
        if len(indices) == 0:
            return []
        
        filtered_matches = []
        for i in indices.flatten():
            filtered_matches.append(matches[i])
        
        return filtered_matches
    
    def detect_currencies(self, frame):
        """Detect all currencies in the frame using all templates"""
        processed = self.preprocess_image(frame)
        detected = []
        
        # Try all templates for each denomination
        for denom, template_list in self.templates.items():
            if not template_list:  # Skip if no templates for this denomination
                continue
            
            all_matches = []
            for template in template_list:
                matches = self.detect_currency_template_matching(processed, template, threshold=0.6)
                all_matches.extend(matches)
            
            # Apply non-max suppression across all matches for this denomination
            if all_matches:
                # Get dimensions from first template (templates should be similar size)
                h, w = template_list[0].shape[:2]
                matches = self.non_max_suppression(all_matches, h, w)
                
                for match in matches:
                    pt = match['point']
                    scale = match['scale']
                    # Use first template to get dimensions (they should be similar)
                    h, w = template_list[0].shape[:2]
                    
                    detected.append({
                        'denomination': denom,
                        'value': self.denominations[denom],
                        'bbox': (
                            pt[0],
                            pt[1],
                            int(w * scale),
                            int(h * scale)
                        ),
                        'confidence': match['confidence']
                    })
        
        return detected
    
    def detect_currencies_orb(self, frame):
        """Alternative detection using ORB feature matching (more robust)"""
        processed = self.preprocess_image(frame)
        detected = []
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Find keypoints and descriptors for the frame once
        kp2, des2 = orb.detectAndCompute(processed, None)
        if des2 is None:
            return detected
        
        # Try all templates for each denomination
        for denom, template_list in self.templates.items():
            if not template_list:  # Skip if no templates for this denomination
                continue
            
            best_match = None
            best_confidence = 0
            
            # Try each template and keep the best match
            for template in template_list:
                # Find keypoints and descriptors for template
                kp1, des1 = orb.detectAndCompute(template, None)
                
                if des1 is None:
                    continue
                
                # Match features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
                
                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                # Need at least 10 good matches
                if len(good_matches) >= 10:
                    confidence = len(good_matches) / 100.0
                    
                    # Keep track of best match
                    if confidence > best_confidence:
                        # Get matched points
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        # Find homography
                        try:
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if M is not None:
                                h, w = template.shape[:2]
                                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                                dst = cv2.perspectiveTransform(pts, M)
                                
                                # Calculate bounding box
                                x_coords = [point[0][0] for point in dst]
                                y_coords = [point[0][1] for point in dst]
                                x, y = int(min(x_coords)), int(min(y_coords))
                                w_box, h_box = int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))
                                
                                best_match = {
                                    'denomination': denom,
                                    'value': self.denominations[denom],
                                    'bbox': (x, y, w_box, h_box),
                                    'confidence': confidence
                                }
                                best_confidence = confidence
                        except:
                            pass
            
            # Add best match if found
            if best_match:
                detected.append(best_match)
        
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{denom} som ({conf:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            total_sum += value
        
        # Draw total sum
        sum_text = f"Total: {total_sum} som"
        cv2.putText(frame, sum_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Draw count
        count_text = f"Items: {len(detections)}"
        cv2.putText(frame, count_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        return frame, total_sum


def main():
    """Main function to run currency detection from camera"""
    print("=" * 60)
    print("Kyrgyz Som Currency Detection System")
    print("=" * 60)
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press SPACE to pause/resume detection")
    print("=" * 60 + "\n")
    
    # Initialize detector
    try:
        detector = CurrencyDetector(currencies_folder="currencies")
        total_templates = sum(len(templates) for templates in detector.templates.values())
        print(f"\nLoaded {total_templates} templates from {len([d for d in detector.templates.values() if d])} denomination folders!\n")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        print("Make sure your camera is connected and not being used by another application.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Camera initialized. Starting detection...\n")
    
    paused = False
    use_orb = True  # Toggle between template matching and ORB
    
    frame_count = 0
    
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
                # Detect currencies (try ORB first, fallback to template matching)
                if use_orb:
                    detections = detector.detect_currencies_orb(frame)
                    if len(detections) == 0:
                        detections = detector.detect_currencies(frame)
                else:
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
            
            # Add method indicator
            method_text = "Method: ORB" if use_orb else "Method: Template Matching"
            cv2.putText(display_frame, method_text, 
                       (10, display_frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
            elif key == ord('m'):
                use_orb = not use_orb
                print(f"Switched to {'ORB' if use_orb else 'Template Matching'} method")
            
            frame_count += 1
    
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

