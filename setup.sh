#!/bin/bash
# Setup script for Currency Detection System

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Setting up Currency Detection System"
echo "=========================================="
echo ""

# Remove old/incomplete venv if it exists
if [ -d "venv" ]; then
    if [ ! -f "venv/bin/activate" ]; then
        echo "Removing incomplete virtual environment..."
        rm -rf venv
    else
        echo "✓ Virtual environment already exists"
        echo ""
        echo "Activating virtual environment..."
        source venv/bin/activate
        
        echo ""
        echo "Upgrading pip..."
        pip install --upgrade pip --quiet
        
        echo "Installing/updating requirements..."
        pip install -r requirements.txt
        
        echo ""
        echo "=========================================="
        echo "Setup complete!"
        echo "=========================================="
        echo ""
        echo "To run the program:"
        echo "  ./run.sh"
        echo ""
        echo "Or manually:"
        echo "  source venv/bin/activate"
        echo "  python main.py"
        echo ""
        exit 0
    fi
fi

# Create new virtual environment
echo "Creating virtual environment..."
if ! python3 -m venv venv; then
    echo ""
    echo "ERROR: python3-venv is not installed."
    echo ""
    echo "Please install it first by running:"
    echo "  sudo apt install python3.12-venv"
    echo ""
    echo "Or if you have a different Python version, use:"
    echo "  sudo apt install python3-venv"
    echo ""
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "Installing requirements (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To run the program:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""

