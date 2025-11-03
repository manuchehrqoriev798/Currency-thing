# Complete Setup and Installation Instructions

## Step 1: Install All Dependencies

### Option A: Automatic Installation (Linux/Mac)
```bash
chmod +x install_all.sh
./install_all.sh
```

### Option B: Manual Installation
```bash
# Install Python packages
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Option C: Using Python Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 2: Prepare Your Currency Images

Your folder structure should look like this:
```
currencies/
├── 20/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (add as many as you want!)
├── 50/
│   └── ... (50 som images)
├── 100/
│   └── ... (100 som images)
├── 500/
│   └── ... (500 som images)
├── 1000/
│   └── ... (1000 som images)
└── 5000/
    └── ... (5000 som images)
```

**Important:**
- Put at least 2-3 images per denomination folder for good results
- More images = better training = better detection!
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Images should show the full currency clearly

---

## Step 3: Train the Model

**This step reads all images from the folders and trains a machine learning model.**

```bash
python3 train_model.py
```

### What happens during training:
1. ✅ Loads all images from `currencies/` folders
2. ✅ Extracts features from each image
3. ✅ Creates training dataset
4. ✅ Trains a Random Forest classifier
5. ✅ Saves the trained model to:
   - `currency_model.pkl` (the trained model)
   - `currency_scaler.pkl` (feature normalizer)
   - `currency_mappings.pkl` (label mappings)

### Expected output:
```
============================================================
Loading training data from folders...
============================================================

  Processing 20 som (2 images)...
    ✓ Loaded 4 samples for 20 som

  Processing 50 som (1 images)...
    ✓ Loaded 2 samples for 50 som
...

Training model...
Training Accuracy: 95.50%
Test Accuracy: 92.00%

✓ Model saved successfully!
```

**⚠️ IMPORTANT:** You must run training first before detection will work!

---

## Step 4: Run Detection (After Training)

**This step uses your trained model to detect currencies from camera.**

```bash
python3 detect_currency.py
```

### What happens:
1. ✅ Loads the trained model
2. ✅ Opens your camera
3. ✅ Detects currencies in real-time
4. ✅ Calculates total sum
5. ✅ Displays results with bounding boxes

### Controls:
- **`q`** - Quit the application
- **`s`** - Save current frame with detections
- **`SPACE`** - Pause/Resume detection

---

## Step 5: Troubleshooting

### Problem: "Model not found"
**Solution:** Run `python3 train_model.py` first!

### Problem: "No module named 'sklearn'"
**Solution:** 
```bash
pip3 install scikit-learn
```

### Problem: Camera doesn't open
**Solutions:**
- Make sure no other app is using the camera
- Check camera permissions
- Try different camera index: change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in detect_currency.py

### Problem: Poor detection accuracy
**Solutions:**
- Add more training images (at least 3-5 per denomination)
- Ensure good lighting when using camera
- Make sure currencies are clearly visible
- Place currencies on contrasting background

### Problem: "No training data found"
**Solutions:**
- Check that `currencies/` folder exists
- Make sure each subfolder (20, 50, 100, etc.) contains image files
- Verify image formats are supported (.jpg, .jpeg, .png, .bmp)

---

## Complete Workflow Summary

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Verify setup (optional)
python3 setup_check.py

# 3. Train model (REQUIRED - do this first!)
python3 train_model.py

# 4. Run detection
python3 detect_currency.py
```

---

## File Structure After Training

```
Currency thing/
├── train_model.py          # Training script (run once)
├── detect_currency.py      # Detection script (run to detect)
├── install_all.sh          # Installation script
├── requirements.txt        # Python dependencies
├── currencies/             # Your currency images
│   ├── 20/
│   ├── 50/
│   └── ...
├── currency_model.pkl      # Trained model (created after training)
├── currency_scaler.pkl     # Feature scaler (created after training)
└── currency_mappings.pkl   # Label mappings (created after training)
```

---

## Quick Reference

| Step | Command | When to Run |
|------|---------|-------------|
| Install | `pip3 install -r requirements.txt` | Once, at the beginning |
| Train | `python3 train_model.py` | Once, before first use |
| Detect | `python3 detect_currency.py` | Every time you want to detect |

**Remember:** You only need to train once (or whenever you add new images to folders)!

---

## Need Help?

1. Run `python3 setup_check.py` to verify your setup
2. Check that all folders have images
3. Make sure training completed successfully
4. Verify camera is working

---

**Good luck with your project! 🎉**

