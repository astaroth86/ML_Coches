FROM python:3.12.2-slim-bookworm

COPY requirements.txt .
RUN pip3 install --default-timeout=1000 -r requirements.txt
RUN mkdir -p app

COPY ./App app
COPY ./data data
COPY ./model model

WORKDIR app

EXPOSE 5000

CMD ["python3", "app.py", "--port", "5000"]
