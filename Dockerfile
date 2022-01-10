FROM python:3.9-slim

EXPOSE 8501
WORKDIR /app

COPY . .
RUN pip install -r .docker/requirements.txt
CMD streamlit run src/app.py
