# ✅ Use official slim Python base image
FROM python:3.10-slim

# ✅ Set working directory
WORKDIR /app

# ✅ Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ✅ Copy requirements file early for Docker caching
COPY requirements.txt .

# ✅ Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# ✅ Download SpaCy model
RUN python -m spacy download en_core_web_sm

# ✅ Copy entire project into the container
COPY . .

# ✅ Fix permissions
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

USER appuser

# ✅ Expose FastAPI port
EXPOSE 5000

# ✅ Add current working dir to PYTHONPATH to avoid import errors
ENV PYTHONPATH="${PYTHONPATH}:/app"

# ✅ Launch Gunicorn with Uvicorn worker
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:5000", "--timeout", "120", "main:app"]
