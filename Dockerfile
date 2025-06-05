FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Cài đặt phiên bản Gradio cụ thể và các dependencies
RUN pip install --no-cache-dir gradio==3.50.2
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# The API key should be passed at runtime
ENV OPENROUTER_API_KEY=""
ENV SITE_URL="http://localhost:7860"
ENV SITE_NAME="Speech Evaluation App"

# Create data directory for any persistent storage needs
RUN mkdir -p /app/data

# Download the model during build to avoid downloading at runtime
RUN python -c "from model import download_and_initialize_model; download_and_initialize_model()"

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]