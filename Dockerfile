FROM python:3.13-slim

COPY .docker/requirements.txt /
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8501
ENTRYPOINT ["python"]
