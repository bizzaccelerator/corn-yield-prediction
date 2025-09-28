FROM python:3.10-slim

RUN pip install --upgrade pip
RUN pip install mlflow[extras] psycopg2-binary gunicorn google-cloud-storage

WORKDIR /app
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE $PORT
ENTRYPOINT ["./entrypoint.sh"]
