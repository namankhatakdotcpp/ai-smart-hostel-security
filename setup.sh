#!/bin/bash

echo "Setting up AI Smart Hostel Security..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete."