FROM python:3.12-slim

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
