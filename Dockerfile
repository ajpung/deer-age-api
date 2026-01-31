FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libxcb1 libx11-6 libxext6 libsmo libiceo libgl1 libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "300", "--workers", "1"]