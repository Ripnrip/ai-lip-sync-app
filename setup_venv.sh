#!/bin/bash
set -e

# Create .venv if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "\n✅ .venv setup complete! To activate later: source .venv/bin/activate"
echo "To run the app: streamlit run app.py" 