# Kyrgyz Som Currency Detection System

A real-time currency detection system that uses your camera to detect Kyrgyz Som banknotes, classify their denominations, and calculate the total sum.

## Features

- 📷 Real-time camera-based currency detection
- 🔍 Multiple detection methods (Template Matching + ORB Feature Matching)
- 💰 Automatic denomination classification (20, 50, 100, 500, 1000, 5000 som)
- 📊 Real-time total sum calculation
- 🎯 Visual feedback with bounding boxes and labels
- 📸 Save detected frames

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Organize currency template images in folders:**
   ```
   currencies/
   ├── 20/        (put all 20 som images here)
   ├── 50/        (put all 50 som images here)
   ├── 100/       (put all 100 som images here)
   ├── 500/       (put all 500 som images here)
   ├── 1000/      (put all 1000 som images here)
   └── 5000/      (put all 5000 som images here)
   ```
   
   You can add multiple images (jpeg, jpg, png, bmp) to each folder for better detection accuracy!

## Usage

1. **Connect your camera** to your computer

2. **Run the detection system:**
   ```bash
   python currency_detector.py
   ```

3. **Controls:**
   - `q` - Quit the application
   - `s` - Save current frame with detections
   - `SPACE` - Pause/Resume detection
   - `m` - Switch between detection methods (ORB/Template Matching)

## How It Works

### Detection Methods

1. **ORB Feature Matching (Default)**
   - Uses ORB (Oriented FAST and Rotated BRIEF) keypoint detection
   - More robust to rotation and scale changes
   - Better for varying lighting conditions

2. **Template Matching**
   - Multi-scale template matching approach
   - Fast and effective for consistent conditions
   - Fallback method if ORB finds no matches

### Image Processing Pipeline

1. **Preprocessing:**
   - Grayscale conversion
   - Histogram equalization for contrast enhancement
   - Gaussian blur for noise reduction

2. **Detection:**
   - Feature extraction and matching
   - Bounding box generation
   - Non-maximum suppression to remove duplicates

3. **Classification:**
   - Matches detected regions with template images
   - Determines denomination based on best match

4. **Visualization:**
   - Draws colored bounding boxes for each detection
   - Labels each currency with denomination and confidence
   - Displays total sum and item count

## Troubleshooting

### Camera not working?
- Make sure no other application is using the camera
- Try disconnecting and reconnecting the camera
- Check camera permissions in your system settings

### Poor detection accuracy?
- Ensure good lighting conditions
- Place currencies on a contrasting background
- Hold currencies flat and clearly visible
- Try adjusting the threshold values in the code

### No templates loaded?
- Verify the `currencies/` folder exists with subfolders for each denomination (20, 50, 100, 500, 1000, 5000)
- Make sure each folder contains at least one image file (.jpeg, .jpg, .png, or .bmp)
- You can add multiple images to each folder for better detection
- Run `python setup_check.py` to verify your folder structure

## Project Structure

```
Currency thing/
├── currency_detector.py  # Main detection script
├── setup_check.py       # Setup verification script
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── project_description.txt  # Detailed project description
└── currencies/          # Currency template folders
    ├── 20/              # 20 som templates (multiple images allowed)
    ├── 50/              # 50 som templates
    ├── 100/             # 100 som templates
    ├── 500/             # 500 som templates
    ├── 1000/            # 1000 som templates
    └── 5000/            # 5000 som templates
```

## Course Alignment

This project implements concepts from Digital Image Processing (COMP 3041):
- Image enhancement (histogram equalization)
- Edge detection and feature extraction
- Object detection and recognition
- Image segmentation techniques
- Real-time image processing

## Future Improvements

- [ ] Support for coin detection
- [ ] Machine learning-based classification
- [ ] Multiple currency support
- [ ] Export results to file
- [ ] Batch processing for saved images

## License

Educational project for UCA Digital Image Processing course.

