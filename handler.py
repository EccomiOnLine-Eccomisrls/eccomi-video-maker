from io import BytesIO
import os
import tempfile
import requests
import runpod
from supabase import Client, create_client
import torch

print("--> [INIT] Avvio dello script handler.py...", flush=True)

# 1. Variabili d'ambiente e Client Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BASE_SUPABASE_URL = SUPABASE_URL.rstrip("/")

if not BASE_SUPABASE_URL or not SUPABASE_KEY:
    print(
        "⚠️ WARNING: SUPABASE_URL o SUPABASE_KEY non impostate!", flush=True
    )

supabase: Client = create_client(BASE_SUPABASE_URL, SUPABASE_KEY)

# Import MoviePy per il montaggio automatico
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    ConcatenateVideoClips,
    VideoFileClip,
)

# 2. Caricamento del modello AI
print("--> [MODEL] Caricamento di CogVideoX-5b-I2V in corso...", flush=True)

try:
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print(
        "--> [MODEL] Modello caricato ed ottimizzato con successo!", flush=True
    )

except Exception as e:
    print(f"❌ [MODEL ERROR] Impossibile caricare il modello: {e}", flush=True)
    raise e


def download_image(url: str):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    from PIL import Image

    return Image.open(BytesIO(response.content)).convert("RGB")


def download_file(url: str, output_path: str):
    """Scarica file audio locali o da URL pubblico"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def generate_single_clip(image, prompt: str) -> str:
    """Genera uno spezzone di video con CogVideoX"""
    print(f"--> [AI GENERATION] Genero clip per: '{prompt}'", flush=True)
    enhanced_prompt = f"{prompt}, cinematic lighting, photorealistic, 4k resolution, smooth motion, high detail"

    frames = pipe(
        image=image,
        prompt=enhanced_prompt,
        num_frames=49,  # ~4 secondi di clip base
        num_inference_steps=35,
        guidance_scale=6.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).frames[0]

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    # Esportiamo a 24 FPS per rendere il movimento ultra-fluido
    export_to_video(frames, temp_file.name, fps=24)
    return temp_file.name


def handler(event):
    print("--- 🚀 Nuova richiesta Spot Pubblicitario Ricevuta! ---", flush=True)
    job_input = event.get("input", {})
    job_id = event.get("id", "test_job")

    image_url = job_input.get("image_url")
    voice_audio_url = job_input.get(
        "voice_audio_url"
    )  # Traccia audio con la voce del personaggio
    music_audio_url = job_input.get(
        "music_audio_url"
    )  # Traccia musicale di sottofondo

    # Prompts per le varie scene che compongono lo spot da 20-30s
    scenes_prompts = job_input.get(
        "scenes_prompts",
        [
            "Character standing heroically in front of city skyline, looking at camera",
            "Character pointing dynamically to a glowing hologram product screen",
            "Character smiling and giving a thumb up in a high tech office, dynamic movement",
            "Cinematic dramatic zoom out, character holding a badge with logo",
        ],
    )

    if not image_url:
        return {"error": "Nessuna image_url fornita."}

    generated_clip_paths = []
    temp_audio_files = []

    try:
        init_image = download_image(image_url)

        # 1. GENERAZIONE MULTI-SCENA (Loop per coprire la durata dello spot)
        for idx, scene_prompt in enumerate(scenes_prompts):
            print(
                f"--> Generating Scene {idx+1}/{len(scenes_prompts)}...",
                flush=True,
            )
            clip_path = generate_single_clip(init_image, scene_prompt)
            generated_clip_paths.append(clip_path)

        # 2. MONTAGGIO VIDEO (MoviePy)
        print("--> [MONTAGGIO] Unisco le clip generate...", flush=True)
        video_clips = [VideoFileClip(p) for p in generated_clip_paths]
        final_video_base = concatenate_videoclips(
            video_clips, method="compose"
        )

        audio_tracks = []

        # 3. GESTIONE AUDIO (Voce + Musica)
        if voice_audio_url:
            voice_path = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ).name
            download_file(voice_audio_url, voice_path)
            temp_audio_files.append(voice_path)

            voice_clip = AudioFileClip(voice_path)
            audio_tracks.append(voice_clip)

        if music_audio_url:
            music_path = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ).name
            download_file(music_audio_url, music_path)
            temp_audio_files.append(music_path)

            music_clip = (
                AudioFileClip(music_path)
                .volumex(0.15)
                .set_duration(final_video_base.duration)
            )
            audio_tracks.append(music_clip)

        # Se sono stati forniti audio, mixali e inseriscili nel video
        if audio_tracks:
            print("--> [AUDIO] Mixaggio traccia audio e musica...", flush=True)
            final_audio = CompositeAudioClip(audio_tracks)
            final_video = final_video_base.set_audio(final_audio)
        else:
            final_video = final_video_base

        # 4. ESPORTAZIONE E RENDER FINALE
        output_spot_path = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ).name
        print("--> [RENDER] Esportazione spot promozionale...", flush=True)
        final_video.write_videofile(
            output_spot_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
        )

        # 5. UPLOAD SU SUPABASE BUCKET 'videos'
        print("--> [UPLOAD] Caricamento spot finale su Supabase...", flush=True)
        object_path = f"{job_id}_spot.mp4"

        with open(output_spot_path, "rb") as f:
            supabase.storage.from_("videos").upload(
                path=object_path,
                file=f,
                file_options={"content-type": "video/mp4", "upsert": "true"},
            )

        public_video_url = supabase.storage.from_("videos").get_public_url(
            object_path
        )

        # Pulizia file temporanei
        for p in generated_clip_paths + temp_audio_files + [output_spot_path]:
            if os.path.exists(p):
                os.remove(p)

        print("✅ Spot Pubblicitario creato con successo!", flush=True)
        return {"spot_url": public_video_url}

    except Exception as e:
        print(f"❌ Errore durante la creazione dello spot: {e}", flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
