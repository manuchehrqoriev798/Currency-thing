"""
Training script for Currency Detection System
Trains a model on images from the currencies/ folder structure
"""

import cv2
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class CurrencyTrainer:
    def __init__(self, currencies_folder="currencies"):
        """Initialize the trainer"""
        self.currencies_folder = currencies_folder
        self.denominations = {
            '20': 20,
            '50': 50,
            '100': 100,
            '500': 500,
            '1000': 1000,
            '5000': 5000
        }
        self.denom_to_label = {denom: idx for idx, denom in enumerate(self.denominations.keys())}
        self.label_to_denom = {idx: denom for denom, idx in self.denom_to_label.items()}
        
    def extract_features(self, image):
        """Extract features from an image"""
        features = []
        
        # Resize image to standard size
        image_resized = cv2.resize(image, (200, 100))
        
        # 1. Color histogram features (RGB)
        if len(image_resized.shape) == 3:
            for i in range(3):
                hist = cv2.calcHist([image_resized], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
        else:
            hist = cv2.calcHist([image_resized], [0], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # 2. Texture features (LBP - Local Binary Pattern approximation)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) if len(image_resized.shape) == 3 else image_resized
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.mean(edges))
        features.append(np.std(edges))
        
        # 3. Statistical features
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.min(gray))
        features.append(np.max(gray))
        
        # 4. Shape features (aspect ratio, area)
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
        
        # 5. Histogram of Oriented Gradients (HOG) simplified
        hog_features = self.compute_hog_simple(gray)
        features.extend(hog_features)
        
        return np.array(features)
    
    def compute_hog_simple(self, gray):
        """Compute simplified HOG features"""
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and direction
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 360
        
        # Create histogram of orientations (8 bins)
        hist, _ = np.histogram(angle, bins=8, range=(0, 360), weights=magnitude)
        return hist.tolist()
    
    def load_training_data(self):
        """Load all images from folders and extract features"""
        print("=" * 60)
        print("Loading training data from folders...")
        print("=" * 60)
        
        X = []  # Features
        y = []  # Labels
        image_extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        
        for denom in self.denominations.keys():
            denom_folder = os.path.join(self.currencies_folder, denom)
            label = self.denom_to_label[denom]
            
            if not os.path.exists(denom_folder):
                print(f"  ⚠ Folder not found: {denom_folder}")
                continue
            
            files = os.listdir(denom_folder)
            image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
            
            if len(image_files) == 0:
                print(f"  ⚠ No images in {denom_folder}")
                continue
            
            print(f"\n  Processing {denom} som ({len(image_files)} images)...")
            count = 0
            
            for filename in image_files:
                file_path = os.path.join(denom_folder, filename)
                image = cv2.imread(file_path)
                
                if image is None:
                    print(f"    ✗ Failed to load: {filename}")
                    continue
                
                # Extract features
                features = self.extract_features(image)
                X.append(features)
                y.append(label)
                count += 1
                
                # Data augmentation: flip horizontally
                flipped = cv2.flip(image, 1)
                features_flipped = self.extract_features(flipped)
                X.append(features_flipped)
                y.append(label)
                count += 1
            
            print(f"    ✓ Loaded {count} samples for {denom} som")
        
        if len(X) == 0:
            raise ValueError("No training data found! Please add images to currency folders.")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n{'=' * 60}")
        print(f"Total samples loaded: {len(X)}")
        print(f"Features per sample: {X.shape[1]}")
        print(f"Number of classes: {len(self.denominations)}")
        print(f"{'=' * 60}\n")
        
        return X, y
    
    def train(self, X, y):
        """Train the classifier"""
        print("=" * 60)
        print("Training model...")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalize features
        print("\nNormalizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest (good for image classification)
        print("\nTraining Random Forest classifier...")
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = classifier.score(X_train_scaled, y_train)
        test_score = classifier.score(X_test_scaled, y_test)
        
        print(f"\nTraining Accuracy: {train_score * 100:.2f}%")
        print(f"Test Accuracy: {test_score * 100:.2f}%")
        
        return classifier, scaler
    
    def save_model(self, classifier, scaler, model_path="currency_model.pkl", scaler_path="currency_scaler.pkl"):
        """Save the trained model"""
        print(f"\n{'=' * 60}")
        print("Saving model...")
        print(f"{'=' * 60}")
        
        joblib.dump(classifier, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save label mappings
        mappings = {
            'denom_to_label': self.denom_to_label,
            'label_to_denom': self.label_to_denom,
            'denominations': self.denominations
        }
        
        with open('currency_mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        print(f"  ✓ Model saved to: {model_path}")
        print(f"  ✓ Scaler saved to: {scaler_path}")
        print(f"  ✓ Mappings saved to: currency_mappings.pkl")
        print(f"\n{'=' * 60}")
        print("Training completed successfully!")
        print(f"{'=' * 60}\n")


def main():
    """Main training function"""
    print("\n" + "=" * 60)
    print("CURRENCY DETECTION - MODEL TRAINING")
    print("=" * 60 + "\n")
    
    try:
        # Initialize trainer
        trainer = CurrencyTrainer(currencies_folder="currencies")
        
        # Load training data
        X, y = trainer.load_training_data()
        
        # Train model
        classifier, scaler = trainer.train(X, y)
        
        # Save model
        trainer.save_model(classifier, scaler)
        
        print("\n✅ Training complete! You can now run detect_currency.py")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

