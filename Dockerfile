# Immagine ufficiale RunPod ottimizzata per AI e GPU Nvidia
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Dipendenze di sistema per immagini, video e audio.
# I font DejaVu servono al compositor deterministico di logo/CTA/testi.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libjpeg-dev \
    libpng-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Torch compatibile con diffusers 0.33+
RUN pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Manteniamo il vecchio handler nel repository come rollback,
# ma l'immagine commerciale usa il nuovo worker.
COPY handler_commercial.py .

CMD [ "python", "-u", "handler_commercial.py" ]
