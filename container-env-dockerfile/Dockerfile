FROM python:3.11
RUN apt update && apt install -y python3-venv gcc
RUN apt-get install -y python3-dev build-essential
RUN pip install -U pip
RUN pip install scikit-learn pandas numpy mlflow boto3 kserve   