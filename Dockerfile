# Use the official Python 3.9 slim image as the base
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files and to buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Copy the download_models.py script into the container
COPY download_models.py .

# Pre-download models by running the download_models.py script
RUN python download_models.py

# Expose port 8000 to allow external access to the FastAPI server
EXPOSE 8000

# Define the command to run the FastAPI server using Uvicorn
CMD ["python", "server.py"]
