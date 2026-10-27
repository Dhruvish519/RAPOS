# RAPOS AI service - includes Tesseract so Chart Vision can read price axes.
# On Render: New > Web Service > "Docker" runtime, root directory = backend
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
