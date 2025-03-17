FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Explicitly install websockets
RUN pip install websockets>=11.0.3

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8501
ENV RENDER=true

# Create necessary directories
RUN mkdir -p models database

# Expose the port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "hypersyncdt/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
