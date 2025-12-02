# Imagen oficial de TensorFlow con CPU (ya trae tensorflow y tensorflow.keras)
FROM tensorflow/tensorflow:2.15.0

# Usar Keras legacy compatible con modelos .h5
ENV TF_USE_LEGACY_KERAS=1

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias, ignorando el blinker viejo que viene en la imagen
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

# Copiar el resto del proyecto
COPY . .

EXPOSE 8050

CMD ["python", "app.py"]