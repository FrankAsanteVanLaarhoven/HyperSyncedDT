FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && apt-get clean

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8501
ENV RENDER=true

# Create necessary directories
RUN mkdir -p models database

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 