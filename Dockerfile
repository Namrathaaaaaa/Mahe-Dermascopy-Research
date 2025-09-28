FROM python:3.12-slim

WORKDIR /app

# Install only the necessary system libs for OpenCV & matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
