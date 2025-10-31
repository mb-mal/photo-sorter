# Multi-stage build for PhotoSorter
FROM python:3.13-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.13-slim

# Install runtime dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY photo_sorter.py .
COPY requirements.txt .

# Create directories for data
RUN mkdir -p /app/img /app/sorted_photos /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Default command
ENTRYPOINT ["python", "photo_sorter.py"]
CMD ["--source", "/app/img", "--destination", "/app/sorted_photos", "--database", "/app/data/photo_analysis.sqlite"]

