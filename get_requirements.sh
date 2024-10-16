#!/bin/bash
# extract_requirements.sh

# Check if virtual environment is active
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "No virtual environment is currently active."
    echo "Please activate your virtual environment and run this script again."
    exit 1
fi

# Extract requirements
pip freeze > requirements.txt

echo "Requirements have been extracted to requirements.txt"
