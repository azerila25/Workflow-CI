FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn \
    pandas

COPY . .

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/app/model", "--host", "0.0.0.0", "--port", "8080", "--no-conda"]