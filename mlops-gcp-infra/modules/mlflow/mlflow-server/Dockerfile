# Dockerfile
FROM python:3.10-slim

# Install dependencies
RUN pip install --upgrade pip
RUN pip install mlflow[extras] psycopg2-binary gunicorn google-cloud-storage

# Create app directory
WORKDIR /app
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Cloud Run usa PORT environment variable
EXPOSE $PORT
ENTRYPOINT ["./entrypoint.sh"]