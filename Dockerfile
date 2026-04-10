FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Clone the Kronos model source code from GitHub
RUN git clone https://github.com/shiyu-coder/Kronos /app/kronos

WORKDIR /app/kronos

# Copy our API files into the Kronos directory
COPY requirements.txt ./api_requirements.txt
COPY app.py ./app.py

# Install Python dependencies (CPU-only torch for Render free tier)
RUN pip install --no-cache-dir -r api_requirements.txt

# HuggingFace model cache directory
ENV HF_HOME=/app/hf_cache
ENV PYTHONPATH=/app/kronos
ENV PORT=8000

EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
