# Base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy and install dependencies first
# (Docker caches this layer — speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY mlruns/ ./mlruns/

# Expose port
EXPOSE 8000

# Start the app
CMD ["python", "src/app.py"]
