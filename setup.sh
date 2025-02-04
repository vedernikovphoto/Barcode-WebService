#!/bin/bash

echo "Starting the environment installation..."

# Check if Python 3.10 is installed
if ! python3.10 --version &>/dev/null; then
    echo "Python 3.10 is required but not installed. Please install Python 3.10."
    exit 1
fi

# Install python3.10-venv if not installed
if ! dpkg -l | grep python3.10-venv &>/dev/null; then
    echo "Installing python3.10-venv..."
    sudo apt update
    sudo apt install -y python3.10-venv
fi

# Create virtual environment if it doesn't exist and the activates it
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
fi

source venv/bin/activate

# Install cmake if not installed
if ! command -v cmake &>/dev/null; then
    echo "Installing cmake..."
    sudo apt install -y cmake
fi

pip install --upgrade pip
pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0
pip install --no-cache-dir -r requirements.txt
pip install --no-build-isolation git+https://github.com/ultralytics/yolov5.git

echo "Environment setup complete."
exec bash --rcfile <(echo "source venv/bin/activate")
