#!/bin/bash

# HyperSyncDT Startup Script

# Check if virtual environment exists and create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if not already installed
if ! pip show streamlit > /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting HyperSyncDT application..."
streamlit run frontend/app.py 