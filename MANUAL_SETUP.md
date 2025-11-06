# Manual Setup Instructions

Follow these steps to set up and run the Currency Detection System manually.

## Step 1: Install Python Virtual Environment Support

First, install the required package (requires sudo):

```bash
sudo apt install python3.12-venv
```

Or for a different Python version:

```bash
sudo apt install python3-venv
```

## Step 2: Navigate to Project Directory

```bash
cd ~/Desktop/Currency\ thing
```

Or:

```bash
cd "/home/manu/Desktop/Currency thing"
```

## Step 3: Create Virtual Environment

```bash
python3 -m venv venv
```

This creates a folder called `venv` with a fresh Python environment.

## Step 4: Activate Virtual Environment

```bash
source venv/bin/activate
```

After activation, you should see `(venv)` at the beginning of your terminal prompt, like:
```
(venv) manu@manu:~/Desktop/Currency thing$
```

## Step 5: Upgrade pip (Optional but Recommended)

```bash
pip install --upgrade pip
```

## Step 6: Install All Required Packages

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (deep learning)
- OpenCV (image processing)
- NumPy
- scikit-learn
- And other dependencies

**Note:** This may take several minutes as TensorFlow is a large package.

## Step 7: Run the Program

```bash
python main.py
```

The program will:
1. Delete old model files
2. Train a new model from your currency images
3. Start real-time detection using your camera

## Step 8: Deactivate Virtual Environment (When Done)

When you're finished, you can deactivate the virtual environment:

```bash
deactivate
```

---

## Quick Reference

**Every time you want to run the program:**

```bash
# 1. Navigate to project
cd ~/Desktop/Currency\ thing

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run the program
python main.py

# 4. When done, deactivate (optional)
deactivate
```

**Note:** You only need to create the virtual environment once (Step 3). After that, just activate it each time you want to use the program.



