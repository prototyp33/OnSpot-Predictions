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

# Copy the deployment gateway code
COPY scripts/deployment/gateway ./gateway
COPY scripts/utils ./utils

# Set Python path
ENV PYTHONPATH=/app

# Run the gateway server
CMD ["python", "-m", "gateway.app"] 