FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir evidently

ENV PORT=8080
ENV EVIDENTLY_WORKSPACE=/workspace

EXPOSE 8080

CMD ["evidently", "ui", "--host", "0.0.0.0", "--port", "8080", "--workspace", "/workspace"]
