FROM python:3.10-slim

WORKDIR /app

# Copiar requisitos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto (app.py, models/, listings_cleaned.csv, etc.)
COPY . .

EXPOSE 8050

CMD ["python", "app.py"]