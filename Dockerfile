# Imagen oficial de TensorFlow con CPU (ya trae tensorflow y tensorflow.keras)
FROM tensorflow/tensorflow:2.15.0

# Forzar uso de Keras "legacy" compatible con modelos .h5
ENV TF_USE_LEGACY_KERAS=1

WORKDIR /app

# Instalar dependencias adicionales (Dash, pandas, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto (app.py, models/, csv, etc.)
COPY . .

EXPOSE 8050

CMD ["python", "app.py"]