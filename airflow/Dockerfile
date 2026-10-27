# Airflow image with the project's ML dependencies baked in
FROM apache/airflow:3.3.2-python3.12

USER airflow
COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir -r /tmp/requirements-api.txt mlflow==3.17.2 requests

# the repo is mounted at /opt/airflow/project (see docker-compose.yaml)
ENV PROJECT_ROOT=/opt/airflow/project \
    PYTHONPATH=/opt/airflow/project/src
