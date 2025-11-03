#!/bin/bash

# Installation script for Currency Detection System
# This will install all required dependencies

echo "=========================================="
echo "Currency Detection System - Installation"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3 first."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "⚠️  pip3 not found, trying to install..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

echo "✓ pip3 found: $(pip3 --version)"
echo ""

# Install Python packages
echo "Installing Python packages..."
echo "----------------------------------------"
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure you have images in the currencies/ folders"
echo "2. Run: python3 train_model.py"
echo "3. Run: python3 detect_currency.py"
echo ""

