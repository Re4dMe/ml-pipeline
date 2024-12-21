FROM python:3.9
RUN apt update && apt install -y python3-venv gcc
RUN apt-get install -y python3-dev build-essential

ENV VENV=/.venv/myenv
RUN python3 -m venv ${VENV}
RUN mkdir -p $VENV/src
ENV PATH=$VENV/bin:$PATH
RUN pip install -U pip
RUN pip install psycopg2 mlflow   

# Expose the port that the MLFlow tracking server runs on
EXPOSE 5000
# Default database credentials
ENV DB_USERNAME=admin
ENV DB_PASSWORD=password
ENV DB_URL=mlflow-database:5432
ENV DB_NAME=mlflow-tracking-server-db
ENV ARTIFACT_PATH=/opt/mlflow/backend

ENTRYPOINT  mlflow server \
        --host 0.0.0.0 \
        --default-artifact-root $ARTIFACT_PATH
