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

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

def enhance_prompt(user_prompt: str) -> str:
    base_enhancements = "cinematic lighting, photorealistic, 4k resolution, smooth animation, dynamic movement, highly detailed."
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
        prompt = enhance_prompt(raw_prompt)
        print(f"Prompt ottimizzato: '{prompt}'")

        print(f"Scarico l'immagine da: {image_url}")
        init_image = download_image(image_url)

        print("Generazione video AI in corso...")
        video_frames = pipe(
            image=init_image,
            prompt=prompt,
            num_frames=49,              # Ripristinato a 49 per evitare Out of Memory
            num_inference_steps=40,     # Ridotto leggermente per velocizzare
            guidance_scale=6.0,
            generator=torch.Generator("cuda").manual_seed(42)
        ).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            video_path = temp_video.name
            
        print("Esportazione dei frame in MP4 a 12 FPS...")
        export_to_video(video_frames, video_path, fps=12) # 12 FPS offre un buon bilanciamento

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
        
        os.remove(video_path)
        
        print("✅ Operazione completata!")
        return {"video_url": public_video_url}

    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
