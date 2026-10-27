# RAPOS AI service - includes Tesseract so Chart Vision can read price axes.
# Deployed by render.yaml (Render Blueprint). No PyTorch at runtime: the LSTM
# is served with NumPy (lstm_numpy.py), which keeps RAM within the free plan.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
