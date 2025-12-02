FROM python:3.11-slim

# Opcional: para evitar warnings de buffer cuando se loguea
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos primero requirements
COPY requirements.txt .

# Instalamos pip actualizado, luego NUMPY + TENSORFLOW con versiones compatibles,
# y después el resto de dependencias del proyecto.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "numpy<2,>=1.23.5" "tensorflow==2.15.0" \
    && pip install --no-cache-dir -r requirements.txt

# Ahora copiamos todo el código del proyecto (app.py, models/, csv, etc.)
COPY . .

EXPOSE 8050

CMD ["python", "app.py"]