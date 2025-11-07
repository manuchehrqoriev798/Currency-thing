#!/usr/bin/env python3
"""
main.py - Currency detection using Roboflow API with camera input.

This script connects to your Roboflow trained model and uses your camera
to detect currency denominations (100 and 20 som) in real-time.
"""

import cv2
import os
import sys
import threading
import time
from roboflow import Roboflow

# Fix Qt/wayland issue by forcing X11 backend
os.environ['QT_QPA_PLATFORM'] = 'xcb'


# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class CurrencyDetector:
    @staticmethod
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
    
    def __init__(self, api_key: str, workspace: str, project: str, model_version: int = 1):
        """
        Initialize Roboflow currency detector.
        
        Args:
            api_key: Your Roboflow API key
            workspace: Your Roboflow workspace name
            project: Your Roboflow project name
            model_version: Model version number (default: 1)
        """
        print("🔌 Connecting to Roboflow...")
        try:
            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            
            # Try to find an available model version
            # Prioritize version 5 if requested, then try others
            self.model = None
            versions_to_try = [model_version] + [v for v in range(1, 11) if v != model_version]
            
            print(f"   Looking for model version {model_version}...")
            for v in versions_to_try:
                try:
                    version_obj = project_obj.version(v)
                    test_model = version_obj.model
                    if test_model is not None:
                        self.model = test_model
                        self.model_version = v
                        print(f"✓ Connected to Roboflow successfully!")
                        print(f"✓ Using model version {v}")
                        break
                    else:
                        if v == model_version:
                            print(f"   ⚠️  Version {v} exists but model is not accessible")
                except Exception as e:
                    if v == model_version:
                        error_msg = str(e)
                        if "not found" in error_msg.lower():
                            print(f"   ❌ Version {v} not found")
                        else:
                            print(f"   ⚠️  Version {v} error: {error_msg[:50]}...")
                    continue
            
            if self.model is None:
                print(f"\n❌ No trained models found in project '{project}'")
                if model_version == 5:
                    print(f"\n💡 Version 5 exists but is not accessible via Python SDK.")
                    print(f"   To make version 5 accessible:")
                    print(f"   1. Go to: https://app.roboflow.com/{workspace}/{project}/{model_version}")
                    print(f"   2. Click 'Deploy Model' button")
                    print(f"   3. Or click 'View Model' and ensure it's deployed")
                    print(f"   4. Wait a few minutes for deployment to complete")
                else:
                    print(f"\nPlease:")
                    print(f"  1. Train a model in Roboflow")
                    print(f"  2. Make sure the model is deployed")
                    print(f"  3. Check that workspace '{workspace}' and project '{project}' are correct")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error connecting to Roboflow: {e}")
            print("\nPlease check:")
            print("  1. Your API key is correct")
            print("  2. Your workspace and project names are correct")
            print("  3. You have internet connection")
            sys.exit(1)
    
    def detect_frame(self, frame):
        """
        Detect currencies in a frame.
        
        Args:
            frame: OpenCV frame (numpy array)
        
        Returns:
            List of detections with bounding boxes and predictions
        """
        try:
            # Convert BGR to RGB for Roboflow
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict using Roboflow model
            # Very high confidence threshold to reduce false positives
            # confidence: minimum confidence threshold (0-100)
            # overlap: IoU threshold for NMS (0-100)
            result = self.model.predict(frame_rgb, confidence=85, overlap=30)
            predictions = result.json()
            
            # Filter to only recognize 100 and 20 som
            # Also filter by minimum confidence (85% minimum)
            all_predictions = predictions.get('predictions', [])
            filtered_predictions = [
                pred for pred in all_predictions 
                if pred.get('class', '').strip() in ['100', '20'] 
                and pred.get('confidence', 0) >= 0.85  # 85% minimum confidence
            ]
            
            # Remove overlapping detections - keep only the highest confidence one
            if len(filtered_predictions) > 1:
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
                
                # Remove overlapping detections - but be less aggressive for separate bills
                # Only remove if they're the SAME class and heavily overlapping (same bill detected twice)
                final_predictions = []
                for pred in filtered_predictions:
                    is_overlapping = False
                    for existing in final_predictions:
                        overlap = calculate_overlap(pred, existing)
                        # Only remove if same class AND high overlap (>70%) - means duplicate detection
                        # If different classes or low overlap, they're separate bills
                        if overlap > 0.7 and pred.get('class') == existing.get('class'):
                            if pred['confidence'] > existing['confidence']:
                                final_predictions.remove(existing)
                                final_predictions.append(pred)
                            is_overlapping = True
                            break
                    if not is_overlapping:
                        final_predictions.append(pred)
                
                return final_predictions
            
            return filtered_predictions
        except Exception as e:
            # Only print error occasionally to avoid spam
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            self._error_count += 1
            if self._error_count % 30 == 0:  # Print every 30th error
                print(f"⚠️  Detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """
        Draw detection boxes and labels on frame with exact borders.
        
        Args:
            frame: OpenCV frame
            detections: List of detection dictionaries from Roboflow
        
        Returns:
            Frame with drawn detections
        """
        display_frame = frame.copy()
        
        # Color mapping - mapped to display names (after swap)
        # Note: colors are based on the DISPLAY name, not Roboflow's detection
        colors = {
            '20': (0, 0, 255),      # Red for 20 som (display)
            '100': (0, 255, 0),     # Green for 100 som (display)
        }
        color_names = {
            '20': 'Red',
            '100': 'Green',
        }
        
        for i, det in enumerate(detections, 1):
            # Get bounding box coordinates (exact borders)
            x = int(det['x'] - det['width'] / 2)
            y = int(det['y'] - det['height'] / 2)
            w = int(det['width'])
            h = int(det['height'])
            
            # Get class from Roboflow and map it for display
            roboflow_class = det.get('class', 'Unknown')
            class_name = self.map_class_name(roboflow_class)  # Apply swap mapping
            confidence = det.get('confidence', 0.0)
            color = colors.get(class_name, (255, 255, 255))
            color_name = color_names.get(class_name, 'White')
            
            # Draw bounding box with thicker lines for exact borders
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw label with color name
            label = f"{i}. {class_name} som ({color_name}) - {confidence:.1%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw label background
            cv2.rectangle(display_frame, (x, y - label_size[1] - 15), 
                         (x + label_size[0] + 15, y), color, -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (x + 8, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw corner markers for exact border visibility
            corner_size = 12
            cv2.line(display_frame, (x, y), (x + corner_size, y), color, 3)
            cv2.line(display_frame, (x, y), (x, y + corner_size), color, 3)
            cv2.line(display_frame, (x + w, y), (x + w - corner_size, y), color, 3)
            cv2.line(display_frame, (x + w, y), (x + w, y + corner_size), color, 3)
            cv2.line(display_frame, (x, y + h), (x + corner_size, y + h), color, 3)
            cv2.line(display_frame, (x, y + h), (x, y + h - corner_size), color, 3)
            cv2.line(display_frame, (x + w, y + h), (x + w - corner_size, y + h), color, 3)
            cv2.line(display_frame, (x + w, y + h), (x + w, y + h - corner_size), color, 3)
        
        return display_frame


def find_camera():
    """Find available camera index."""
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return i
        except Exception:
            continue
    return None


def main():
    """Main function to run currency detection with camera."""
    
    # Get API credentials from environment variables or use defaults
    api_key = os.getenv('ROBOFLOW_API_KEY', 'OR7LrKLkb5DbaRn95f5X')
    workspace = os.getenv('ROBOFLOW_WORKSPACE', 'trial-ilic7')  # Workspace from URL
    project = os.getenv('ROBOFLOW_PROJECT', 'main-fddlv')  # Project ID from URL: main-fddlv
    model_version = int(os.getenv('ROBOFLOW_MODEL_VERSION', '2'))  # Using version 2
    
    # Check if credentials are set
    if not api_key or not workspace or not project:
        print("=" * 70)
        print("❌ Roboflow API credentials not found!")
        print("=" * 70)
        print("\nPlease set the following environment variables:")
        print("  export ROBOFLOW_API_KEY='your_api_key_here'")
        print("  export ROBOFLOW_WORKSPACE='your_workspace_name'")
        print("  export ROBOFLOW_PROJECT='your_project_name'")
        print("  export ROBOFLOW_MODEL_VERSION='2'  # Optional, defaults to 2")
        print("\nOr create a .env file with these variables.")
        sys.exit(1)
    
    # Initialize detector
    detector = CurrencyDetector(api_key, workspace, project, model_version)
    
    # Find camera
    print("\n📷 Looking for camera...")
    camera_index = find_camera()
    if camera_index is None:
        print("❌ No camera found. Please connect a camera and try again.")
        sys.exit(1)
    
    print(f"✓ Camera found at index {camera_index}")
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        sys.exit(1)
    
    # Set camera properties for better performance
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce lag
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS if supported
        # Use V4L2 backend for better performance on Linux
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    
    print("\n" + "=" * 70)
    print("🎥 Currency Detection Started")
    print("=" * 70)
    print("Press 'q' to quit")
    print("Press SPACE to pause/resume")
    print()
    
    paused = False
    window_available = True
    latest_detections = []
    latest_frame = None
    detection_lock = threading.Lock()
    running = True
    
    # Stability mechanism: keep history of recent detections to prevent rapid switching
    detection_history = []  # Store last 5 detections
    stable_detections = []  # Current stable detections to display
    history_size = 5  # Number of frames to consider for stability
    
    # Test if we can create a window before starting the loop
    try:
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            cv2.namedWindow('Currency Detection - Roboflow', cv2.WINDOW_NORMAL)
            cv2.imshow('Currency Detection - Roboflow', test_frame)
            cv2.waitKey(1)
    except (cv2.error, Exception) as e:
        window_available = False
        print("\n⚠️  Display window not available. Install GTK libraries:")
        print("   sudo apt-get install -y libgtk2.0-dev pkg-config")
        print("   Then reinstall opencv-python: pip install --upgrade --force-reinstall opencv-python")
        print("\n💡 Detection will still work! Showing results in terminal...")
        print("   (Press Ctrl+C to stop)\n")
    
    # Thread function for detection (runs in background)
    def detection_thread():
        nonlocal latest_detections, latest_frame
        frame_skip = 0
        while running:
            if not paused and latest_frame is not None:
                frame_skip += 1
                # Run detection every 10 frames to keep UI smooth
                if frame_skip >= 10:
                    frame_skip = 0
                    try:
                        detections = detector.detect_frame(latest_frame)
                        with detection_lock:
                            latest_detections = detections
                    except Exception:
                        pass
            time.sleep(0.03)  # Small sleep to prevent CPU spinning
    
    # Start detection thread
    det_thread = threading.Thread(target=detection_thread, daemon=True)
    det_thread.start()
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Could not read frame from camera")
                    break
                
                # Update latest frame for detection thread
                latest_frame = frame.copy()
                
                # Get latest detections (thread-safe)
                with detection_lock:
                    current_detections = latest_detections.copy()
                
                # Stability mechanism: only show detections that are consistent
                # Note: We use original Roboflow class names for matching/stability
                # The class name swap (20<->100) is applied later when displaying results
                if current_detections:
                    # Add current detection to history
                    # Create a signature for this detection (using original Roboflow class names)
                    detection_signature = tuple(sorted([det.get('class', 'Unknown') for det in current_detections]))
                    detection_history.append(detection_signature)
                    
                    # Keep only last N detections
                    if len(detection_history) > history_size:
                        detection_history.pop(0)
                    
                    # Check if we have enough history
                    if len(detection_history) >= 4:
                        # Count occurrences of each signature in recent history
                        recent_history = detection_history[-4:]  # Last 4 frames
                        signature_counts = {}
                        for sig in recent_history:
                            signature_counts[sig] = signature_counts.get(sig, 0) + 1
                        
                        # Find the most common signature (must appear at least 3 times out of 4)
                        most_common = max(signature_counts.items(), key=lambda x: x[1])
                        if most_common[1] >= 3:  # Appeared in at least 3 of last 4 frames
                            stable_signature = most_common[0]
                            # Filter current detections to match stable signature
                            stable_detections = [
                                det for det in current_detections 
                                if det.get('class', 'Unknown') in stable_signature
                                and det.get('confidence', 0) >= 0.85  # Double-check confidence
                            ]
                            # Sort by confidence and take top matches
                            stable_detections = sorted(stable_detections, key=lambda x: x.get('confidence', 0), reverse=True)
                            stable_detections = stable_detections[:len(stable_signature)]
                        else:
                            # Not stable enough - clear detections to prevent false positives
                            stable_detections = []
                    else:
                        # Not enough history yet - don't show anything until stable
                        stable_detections = []
                else:
                    # No detections - clear immediately
                    detection_history.append(())  # Empty detection
                    if len(detection_history) > history_size:
                        detection_history.pop(0)
                    # Clear stable detections immediately when no detection
                    stable_detections = []
                
                # Print stable detections to terminal
                if stable_detections:
                    # Clear previous line and print new detections
                    print("\r" + " " * 80 + "\r", end="")  # Clear line
                    detection_text = " | ".join([
                        f"{detector.map_class_name(det.get('class', 'Unknown'))} som ({det.get('confidence', 0):.1%})"
                        for det in stable_detections
                    ])
                    print(f"💰 Detected: {detection_text}", end="", flush=True)
                else:
                    print("\r" + " " * 80 + "\r", end="")  # Clear line
                
                # Draw stable detections on frame
                display_frame = detector.draw_detections(frame, stable_detections)
                
                # Show frame immediately for smooth playback
                if window_available:
                    try:
                        cv2.imshow('Currency Detection - Roboflow', display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord(' '):
                            paused = not paused
                            if paused:
                                print("⏸️  Paused")
                            else:
                                print("▶️  Resumed")
                    except (cv2.error, Exception):
                        window_available = False
                        print("\n⚠️  Display window closed. Continuing with terminal output...\n")
                else:
                    # Terminal output already handled above with stable_detections
                    time.sleep(0.03)  # Small delay to prevent CPU spinning
            else:
                # When paused, still handle key input
                if window_available:
                    try:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord(' '):
                            paused = False
                            print("▶️  Resumed")
                    except (cv2.error, Exception):
                        pass
                else:
                    time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        running = False  # Stop detection thread
        if det_thread.is_alive():
            det_thread.join(timeout=1.0)
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Ignore if windows can't be destroyed
        print("\n✓ Camera released. Goodbye!")


if __name__ == "__main__":
    main()
