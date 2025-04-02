FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make sure scripts are executable
RUN chmod +x scripts/*.py

# Expose API port
EXPOSE 5000

# Run the API server
CMD ["python", "scripts/deployment/prediction_service.py"] 