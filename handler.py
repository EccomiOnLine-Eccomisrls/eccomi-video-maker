import runpod
import torch
import requests
import tempfile
import os
from io import BytesIO
from PIL import Image
from supabase import create_client, Client

print("--> [INIT] Avvio dello script handler.py...", flush=True)

# 1. Variabili d'ambiente e Client Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BASE_SUPABASE_URL = SUPABASE_URL.rstrip("/")

if not BASE_SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ WARNING: SUPABASE_URL o SUPABASE_KEY non impostate nelle Environment Variables!", flush=True)

# Inizializzazione del client Supabase
supabase: Client = create_client(BASE_SUPABASE_URL, SUPABASE_KEY)

# 2. Caricamento del modello AI
print("--> [MODEL] Caricamento di CogVideoX-5b-I2V in corso...", flush=True)

try:
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        torch_dtype=torch.float16
    )

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print("--> [MODEL] Modello caricato ed ottimizzato con successo!", flush=True)

except Exception as e:
    print(f"❌ [MODEL ERROR] Impossibile caricare il modello: {e}", flush=True)
    raise e


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def enhance_prompt(user_prompt: str) -> str:
    base_enhancements = "cinematic lighting, photorealistic, 4k resolution, smooth animation, dynamic movement, highly detailed."
    return f"{user_prompt}, {base_enhancements}"


def handler(event):
    print("--- 🚀 Nuova richiesta Video Ricevuta! ---", flush=True)
    job_input = event.get("input", {})
    
    image_url = job_input.get("image_url")
    raw_prompt = job_input.get("prompt", "A dynamic professional video of the character")
    job_id = event.get("id", "test_job")

    if not image_url:
        return {"error": "Nessuna image_url fornita."}

    try:
        prompt = enhance_prompt(raw_prompt)
        print(f"--> Prompt ottimizzato: '{prompt}'", flush=True)

        print(f"--> Scarico l'immagine da: {image_url}", flush=True)
        init_image = download_image(image_url)

        print("--> Generazione video AI in corso...", flush=True)
        video_frames = pipe(
            image=init_image,
            prompt=prompt,
            num_frames=49,
            num_inference_steps=35,
            guidance_scale=6.0,
            generator=torch.Generator("cuda").manual_seed(42)
        ).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            video_path = temp_video.name
            
        print("--> Esportazione dei frame in MP4 a 12 FPS...", flush=True)
        export_to_video(video_frames, video_path, fps=12)

        print("--> Caricamento del video su Supabase...", flush=True)
        object_path = f"video_generati/{job_id}.mp4"
        
        # Upload tramite SDK Supabase
        with open(video_path, "rb") as f:
            supabase.storage.from_("inputs").upload(
                path=object_path,
                file=f,
                file_options={"content-type": "video/mp4", "upsert": "true"}
            )
        
        public_video_url = supabase.storage.from_("inputs").get_public_url(object_path)
        
        if os.path.exists(video_path):
            os.remove(video_path)
        
        print("✅ Operazione completata con successo!", flush=True)
        return {"video_url": public_video_url}

    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}", flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
