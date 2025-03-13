#!/bin/bash
cd "$(dirname "$0")"
echo "Starting HyperSyncedDT Application..."

# Check if environment setup is complete
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating a minimal .env file with dummy keys..."
    cat > .env << EOL
PERPLEXITY_API_KEY=dummy_perplexity_key
EDEN_API_KEY=dummy_eden_key
OPENAI_API_KEY=dummy_openai_key
DEBUG_MODE=false
ENVIRONMENT=development
EOL
    echo ".env file created with default values."
fi

echo "You can access the dashboard at http://localhost:8501 when it's ready"

# Set the PYTHONPATH to include the frontend directory
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/frontend"

# Run the application
echo "Starting Streamlit server..."
streamlit run frontend/app.py
