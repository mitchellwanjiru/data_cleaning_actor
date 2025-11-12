# Use official Python image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY .actor/ ./.actor/
COPY key_value_stores/ ./key_value_stores/

# Entrypoint
CMD ["python", "src/main.py"]
