# Use Python 3.12 slim based on Debian Bookworm
FROM python:3.12.1-slim-bookworm

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model file
COPY predict.py model_xgb.bin ./

# Expose port for FastAPI
EXPOSE 9696

# Run the API
CMD ["python", "predict.py"]
