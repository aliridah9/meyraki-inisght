FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Make port 8000 available to the world outside this container
EXPOSE $PORT

# Run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} 