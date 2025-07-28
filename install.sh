#!/bin/bash

echo "Setting up virtual environment for sql-lineage..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip to latest version
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "Installation complete!"
echo "To activate the environment in the future, run:"
echo "source venv/bin/activate"