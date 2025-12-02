FROM python:3.10-slim

# Forzar a TensorFlow a usar el Keras legacy
ENV TF_USE_LEGACY_KERAS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["python", "app.py"]
