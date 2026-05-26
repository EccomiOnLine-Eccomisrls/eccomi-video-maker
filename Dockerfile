# Immagine ufficiale Runpod ottimizzata per AI e schede video Nvidia
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Imposta la cartella di lavoro
WORKDIR /

# Copia e installa i requisiti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il "cervello" del nostro server
COPY handler.py .

# Comando per avviare il server in ascolto perenne
CMD ["python", "-u", "/handler.py"]
