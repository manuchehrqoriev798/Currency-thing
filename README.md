# Kyrgyz Som Currency Detection System

A deep learning-based currency recognition system for Kyrgyz Som banknotes using TensorFlow/Keras with transfer learning.

## Setup Instructions

### Step 1: Install Python Virtual Environment Support

First, you need to install the python3-venv package:

```bash
sudo apt install python3.12-venv
```

Or if you have a different Python version:

```bash
sudo apt install python3-venv
```

### Step 2: Run the Setup Script

Run the setup script to create a virtual environment and install dependencies:

```bash
./setup.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 3: Run the Program

**Easy way (recommended):**
```bash
./run.sh
```

The `run.sh` script automatically:
- Checks if virtual environment exists
- Uses the correct Python from venv
- Installs missing packages if needed
- Runs the program

**Manual way:**
```bash
source venv/bin/activate
python main.py
```

## Features

- **Deep Learning Model**: Uses EfficientNetB0 with transfer learning
- **Heavy Training**: 50 epochs with two-stage training (frozen base + fine-tuning)
- **Data Augmentation**: Heavy augmentation including rotations, shifts, zoom, brightness adjustments
- **Auto-Retrain**: Automatically deletes old models and retrains on every run
- **JPG Only**: Only processes JPG/JPEG image files

## Currency Denominations

The system recognizes the following Kyrgyz Som denominations:
- 20 som
- 50 som
- 100 som
- 200 som
- 500 som
- 1000 som
- 5000 som

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- OpenCV
- NumPy
- scikit-learn

## Using Phone Camera (USB/Type-C Connection)

The system can automatically detect and use your phone camera when connected via USB.

### Setup Your Phone as Webcam

**For Android Phones:**
1. Install a webcam app from Play Store:
   - **DroidCam** (recommended) - Free
   - **iVCam** - Free/Paid
   - Or use Android's built-in webcam feature (Android 14+)

2. Connect your phone to computer via USB Type-C cable

3. Enable USB debugging on your phone:
   - Go to Settings > About Phone
   - Tap "Build Number" 7 times to enable Developer Options
   - Go to Settings > Developer Options
   - Enable "USB Debugging"

4. Open the webcam app on your phone and start the webcam mode

5. Run the camera detection script:
   ```bash
   source venv/bin/activate
   python list_cameras.py
   ```

**For iPhone (Linux/Windows/Mac):**
1. **Option 1: EpocCam** (Recommended for Linux)
   - Download **EpocCam** from App Store (free with paid pro features)
   - Install the EpocCam driver on your Linux computer:
     ```bash
     # Download from: https://www.kinoni.com/epoccam/
     # Or use the free version which works via WiFi/USB
     ```
   - Connect iPhone via USB or WiFi
   - Open EpocCam app on iPhone
   - Camera will appear as a webcam device

2. **Option 2: iVCam** (Alternative)
   - Download **iVCam** from App Store (free with paid features)
   - Install iVCam server on Linux:
     ```bash
     # Download from: https://www.e2esoft.com/ivcam/
     ```
   - Connect iPhone via USB or WiFi
   - Open iVCam app on iPhone
   - Camera will appear as a webcam device

3. **Option 3: Continuity Camera** (Mac only)
   - Built-in feature, no app needed
   - Connect iPhone via USB or wirelessly
   - Camera appears automatically

### Using a Specific Camera

If multiple cameras are detected, you can specify which one to use:

**Method 1: Command line argument**
```bash
python main.py <camera_index>
```
Example: `python main.py 1` to use camera index 1

**Method 2: Environment variable**
```bash
export CAMERA_INDEX=1
python main.py
```

**Method 3: Automatic detection**
- The system will automatically use the last detected camera (usually your phone)
- Run `python list_cameras.py` to see all available cameras

## Usage

1. Place your currency images in the `currencies/` folder:
   ```
   currencies/
   ├── 20/
   │   └── *.jpg
   ├── 50/
   │   └── *.jpg
   ├── 100/
   │   └── *.jpg
   ├── 200/
   │   └── *.jpg
   ├── 500/
   │   └── *.jpg
   ├── 1000/
   │   └── *.jpg
   └── 5000/
       └── *.jpg
   ```

2. Run `main.py` - it will:
   - Delete old model files
   - Train a new model from your images
   - Start real-time detection using your camera

## Controls

- `q` - Quit
- `s` - Save current frame
- `SPACE` - Pause/Resume

