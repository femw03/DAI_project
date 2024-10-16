#!/bin/bash
# setup_venv.sh

# Set variables
VENV_NAME="venv"
REQUIREMENTS_FILE="requirements.txt"

# Create virtual environment
python3.8 -m venv $VENV_NAME

# Activate virtual environment
source $VENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r $REQUIREMENTS_FILE
else
    echo "Requirements file not found: $REQUIREMENTS_FILE"
fi

echo "Virtual environment '$VENV_NAME' created and packages installed."

# PowerShell script (setup_venv.ps1)

# Set variables
$VENV_NAME = "myenv"
$REQUIREMENTS_FILE = "requirements.txt"

# Create virtual environment
python -m venv $VENV_NAME

# Activate virtual environment
& "$VENV_NAME\Scripts\Activate.ps1"

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
if (Test-Path $REQUIREMENTS_FILE) {
    pip install -r $REQUIREMENTS_FILE
}
else {
    Write-Output "Requirements file not found: $REQUIREMENTS_FILE"
}

Write-Output "Virtual environment '$VENV_NAME' created and packages installed."