FROM tensorflow/tensorflow:2.15.0-gpu
# Set working directory
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    libxml2-dev \
    libcairo2-dev \
    pkg-config \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements file
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Install FastAPI directly (if not in requirements)
# RUN pip install --no-cache-dir fastapi
# Copy your application code
COPY . /app
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app
# Default command (will be overridden by AWS Batch)
CMD ["python", "main.py"]