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

# 1. Inizializza Supabase (ambiente Runpod)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Carica il modello AI nella memoria VRAM
print("Caricamento del modello AI in corso...")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.float16
).to("cuda")
print("Modello caricato con successo!")

def download_image(url: str) -> Image.Image:
    """Scarica e converte l'immagine di partenza."""
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def enhance_prompt(user_prompt: str) -> str:
    """Arricchisce il prompt per garantire dettagli cinematici e fluidità."""
    base_enhancements = "cinematic lighting, photorealistic, 4k resolution, smooth high-quality animation, dynamic movement, highly detailed."
    return f"{user_prompt}, {base_enhancements}"

def handler(event):
    print("--- Nuova richiesta Video Ricevuta! ---")
    job_input = event.get("input", {})
    
    image_url = job_input.get("image_url")
    raw_prompt = job_input.get("prompt", "A dynamic professional video of the character")
    job_id = event.get("id", "test_job")

    if not image_url:
        return {"error": "Nessuna image_url fornita."}

    try:
        # 1. Ottimizzazione Prompt
        prompt = enhance_prompt(raw_prompt)
        print(f"Prompt ottimizzato: '{prompt}'")

        # 2. Scarica Immagine
        print(f"Scarico l'immagine da: {image_url}")
        init_image = download_image(image_url)

        # 3. Generazione Video (81 frame per circa 5 secondi di animazione)
        print("Generazione video AI in corso...")
        video_frames = pipe(
            image=init_image,
            prompt=prompt,
            num_frames=81,              # Aumentato per maggiore durata/fluidità
            num_inference_steps=50,
            guidance_scale=6.0,
            generator=torch.Generator("cuda").manual_seed(42)
        ).frames[0]

        # 4. Esportazione MP4 a 16 FPS (movimento naturale)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            video_path = temp_video.name
            
        print("Esportazione dei frame in MP4 a 16 FPS...")
        export_to_video(video_frames, video_path, fps=16)

        # 5. Upload su Supabase Storage
        print("Caricamento del video su Supabase...")
        object_path = f"video_generati/{job_id}.mp4"
        upload_url = f"{SUPABASE_URL}/storage/v1/object/public/inputs/{object_path}?upsert=true"
        
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
        
        # Pulizia file temporaneo
        os.remove(video_path)
        
        print("✅ Operazione completata!")
        return {"video_url": public_video_url}

    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}")
        return {"error": str(e)}

# 3. Avvio Serverless Runpod
runpod.serverless.start({"handler": handler})

