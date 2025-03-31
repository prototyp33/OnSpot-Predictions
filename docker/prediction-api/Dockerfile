FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY scripts/api ./api
COPY scripts/models ./models
COPY scripts/utils ./utils

# Create directories for model storage
RUN mkdir -p /models/registry /models/production

# Set Python path
ENV PYTHONPATH=/app

# Run the API server
CMD ["python", "-m", "api.app"] 