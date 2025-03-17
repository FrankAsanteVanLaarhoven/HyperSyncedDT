FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && apt-get clean

EXPOSE 8501

# Run the Streamlit app with the correct path
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 