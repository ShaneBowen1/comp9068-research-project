# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY aws_requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r aws_requirements.txt

# COPY the application code
COPY *.py .

# COPY the src directory
COPY src/ src/
