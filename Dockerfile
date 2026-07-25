# Immagine ufficiale RunPod ottimizzata per AI e GPU Nvidia
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Imposta la cartella di lavoro su /app invece della root
WORKDIR /app

# Disabilita il buffering dell'output Python per vedere i log immediatamente
ENV PYTHONUNBUFFERED=1

# Installa le dipendenze di sistema
RUN apt-get update && apt-get install -y libjpeg-dev libpng-dev && rm -rf /var/lib/apt/lists/*

# Copia e installa i requisiti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia lo script handler
COPY handler.py .

# Comando per avviare il server in ascolto
CMD [ "python", "-u", "handler.py" ]
