FROM python:3.11-slim

# Install required system packages for Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Evidently
RUN pip install --no-cache-dir evidently==0.7.12

# Environment variables for Cloud Run
ENV PORT=8080
ENV EVIDENTLY_WORKSPACE=/workspace

# Expose Cloud Run port
EXPOSE 8080

# Start Evidently UI
CMD ["evidently", "ui", "--host", "0.0.0.0", "--port", "8080", "--workspace", "/workspace"]
