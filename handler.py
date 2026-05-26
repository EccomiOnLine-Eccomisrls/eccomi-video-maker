import runpod
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
import requests
import tempfile
import os
from supabase import create_client, Client
from io import BytesIO
from PIL import Image

# Inizializza Supabase (le chiavi le passeremo da Runpod)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Carica il modello AI nella memoria della Scheda Video (VRAM)
# Usiamo float16 per dimezzare il consumo di RAM mantenendo la qualità
print("Caricamento del modello AI in corso...")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.float16
).to("cuda")
print("Modello caricato con successo!")

def download_image(url: str) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def handler(event):
    print("--- Nuova richiesta Video Ricevuta! ---")
    job_input = event.get("input", {})
    
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt", "A dynamic professional video of the product")
    job_id = event.get("id", "test_job")

    if not image_url:
        return {"error": "Nessuna image_url fornita."}

    try:
        # 1. Scarica l'immagine di partenza
        print(f"Scarico l'immagine da: {image_url}")
        init_image = download_image(image_url)

        # 2. Genera il video con l'AI
        print(f"Generazione video in corso per il prompt: '{prompt}'")
        video_frames = pipe(
            image=init_image,
            prompt=prompt,
            num_frames=49, # Numero di frame standard per animazioni fluide
            num_inference_steps=50,
            guidance_scale=6.0,
            generator=torch.Generator("cuda").manual_seed(42)
        ).frames[0]

        # 3. Salva il video in un file temporaneo
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            video_path = temp_video.name
            
        print("Esportazione dei frame in MP4...")
        export_to_video(video_frames, video_path, fps=8)

        # 4. Carica il video su Supabase Storage
        print("Caricamento del video su Supabase...")
        object_path = f"video_generati/{job_id}.mp4"
        upload_url = f"{SUPABASE_URL}/storage/v1/object/inputs/{object_path}?upsert=true"
        
        with open(video_path, "rb") as f:
            video_data = f.read()
            
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
            "Content-Type": "video/mp4",
        }
        
        r_upload = requests.put(upload_url, headers=headers, data=video_data)
        r_upload.raise_for_status()
        
        public_video_url = f"{SUPABASE_URL}/storage/v1/object/public/inputs/{object_path}"
        
        # Pulizia
        os.remove(video_path)
        
        print("✅ Operazione completata!")
        return {"video_url": public_video_url}

    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}")
        return {"error": str(e)}

# Avvia l'ascolto su Runpod
runpod.serverless.start({"handler": handler})
