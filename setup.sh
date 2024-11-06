#!/bin/bash

echo "Creating Virtual Environment..."
python3 -m venv env
source env/bin/activate

echo "Updating pip..."
pip install --upgrade pip

echo "Installing required libraries..."
pip install mne scikit-learn numpy pandas matplotlib seaborn

echo "Installed libraries:"
pip list | grep -E "mne|scikit-learn|numpy|pandas|matplotlib|seaborn"

echo "Configuration complete."
