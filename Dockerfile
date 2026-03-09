FROM python:3.9-slim

WORKDIR /app

# Install libgomp (required by LightGBM)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY mlruns/ ./mlruns/

EXPOSE 8000

CMD ["python", "src/app.py"]