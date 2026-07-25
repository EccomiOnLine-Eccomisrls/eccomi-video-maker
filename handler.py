import os
import tempfile
from io import BytesIO
from PIL import Image
import requests
import runpod
from supabase import Client, create_client
import torch

# Import flessibile per MoviePy (compatibile sia con v1.x che v2.x)
try:
    from moviepy.editor import (
        AudioFileClip,
        CompositeAudioClip,
        VideoFileClip,
        concatenate_videoclips,
    )
except ImportError:
    from moviepy import (
        AudioFileClip,
        CompositeAudioClip,
        VideoFileClip,
        concatenate_videoclips,
    )

print("--> [INIT] Avvio dello script handler.py...", flush=True)

# 1. Variabili d'ambiente e Client Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BASE_SUPABASE_URL = SUPABASE_URL.rstrip("/")

if not BASE_SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ WARNING: SUPABASE_URL o SUPABASE_KEY non impostate!", flush=True)

supabase: Client = create_client(BASE_SUPABASE_URL, SUPABASE_KEY)

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


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def download_file(url: str, output_path: str):
    """Scarica un file generico da URL"""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def extract_or_download_audio(url: str, output_audio_path: str):
    """
    Scarica l'URL fornito:
    - Se è un video (.mov, .mp4, ecc.), estrae automaticamente la traccia audio.
    - Se è già un audio (.mp3, .wav, ecc.), lo salva direttamente.
    """
    temp_download = tempfile.NamedTemporaryFile(delete=False).name
    download_file(url, temp_download)

    video_extensions = [".mov", ".mp4", ".avi", ".mkv", ".webm", ".quicktime"]
    is_video = any(url.lower().endswith(ext) for ext in video_extensions)

    if is_video:
        print(
            f"--> [AUDIO EXTRACT] Rilevato file video da {url}. Estraggo la traccia audio...",
            flush=True,
        )
        video_clip = VideoFileClip(temp_download)
        if video_clip.audio is not None:
            video_clip.audio.write_audiofile(
                output_audio_path, logger=None, fps=44100
            )
            video_clip.close()
        else:
            print(
                "⚠️ WARNING: Il video scaricato non contiene tracce audio!",
                flush=True,
            )
            output_audio_path = None
        if os.path.exists(temp_download):
            os.remove(temp_download)
    else:
        # È già un file audio
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        os.rename(temp_download, output_audio_path)


def generate_single_clip(image: Image.Image, prompt: str) -> str:
    """Genera uno spezzone di video con CogVideoX"""
    print(f"--> [AI GENERATION] Genero clip per: '{prompt}'", flush=True)
    enhanced_prompt = f"{prompt}, cinematic lighting, photorealistic, 4k resolution, smooth motion, high detail"

    frames = pipe(
        image=image,
        prompt=enhanced_prompt,
        num_frames=49,
        num_inference_steps=35,
        guidance_scale=6.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).frames[0]

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    export_to_video(frames, temp_file.name, fps=24)
    return temp_file.name


def handler(event):
    print("--- 🚀 Nuova richiesta Spot Pubblicitario Ricevuta! ---", flush=True)
    job_input = event.get("input", {})
    job_id = event.get("id", "test_job")

    image_url = job_input.get("image_url")
    voice_url = job_input.get("voice_audio_url")
    music_url = job_input.get("music_audio_url")

    scenes_prompts = job_input.get(
        "scenes_prompts",
        [
            "Character standing heroically in front of city skyline, looking dynamically at camera",
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

        # 1. GENERAZIONE SCENE VIDEO
        for idx, scene_prompt in enumerate(scenes_prompts):
            print(
                f"--> Generating Scene {idx+1}/{len(scenes_prompts)}...",
                flush=True,
            )
            clip_path = generate_single_clip(init_image, scene_prompt)
            generated_clip_paths.append(clip_path)

        # 2. MONTAGGIO VIDEO
        print("--> [MONTAGGIO] Unisco le clip generate...", flush=True)
        video_clips = [VideoFileClip(p) for p in generated_clip_paths]
        final_video_base = concatenate_videoclips(
            video_clips, method="compose"
        )

        audio_tracks = []

        # 3. GESTIONE VOCE
        if voice_url:
            voice_audio_path = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ).name
            extract_or_download_audio(voice_url, voice_audio_path)

            if voice_audio_path and os.path.exists(voice_audio_path):
                temp_audio_files.append(voice_audio_path)
                voice_clip = AudioFileClip(voice_audio_path)
                audio_tracks.append(voice_clip)

        # 4. GESTIONE MUSICA DI SOTTOFONDO
        if music_url:
            music_audio_path = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ).name
            extract_or_download_audio(music_url, music_audio_path)

            if music_audio_path and os.path.exists(music_audio_path):
                temp_audio_files.append(music_audio_path)
                music_clip = AudioFileClip(music_audio_path)
                
                # Gestione volume sia per MoviePy v1 che v2
                if hasattr(music_clip, 'volumex'):
                    music_clip = music_clip.volumex(0.15)
                elif hasattr(music_clip, 'with_volume_scaling'):
                    music_clip = music_clip.with_volume_scaling(0.15)

                # Imposta durata
                if hasattr(music_clip, 'set_duration'):
                    music_clip = music_clip.set_duration(final_video_base.duration)
                elif hasattr(music_clip, 'with_duration'):
                    music_clip = music_clip.with_duration(final_video_base.duration)

                audio_tracks.append(music_clip)

        # Mix finale
        if audio_tracks:
            print(
                "--> [AUDIO] Unione traccia vocale e musica di sottofondo...",
                flush=True,
            )
            final_audio = CompositeAudioClip(audio_tracks)
            if hasattr(final_video_base, 'set_audio'):
                final_video = final_video_base.set_audio(final_audio)
            else:
                final_video = final_video_base.with_audio(final_audio)
        else:
            final_video = final_video_base

        # 5. ESPORTAZIONE SPOT COMPLETO
        output_spot_path = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ).name
        print("--> [RENDER] Esportazione spot finale...", flush=True)
        final_video.write_videofile(
            output_spot_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
        )

        # 6. UPLOAD SU SUPABASE
        print("--> [UPLOAD] Caricamento spot su Supabase...", flush=True)
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
            if p and os.path.exists(p):
                os.remove(p)

        print("✅ Spot Pubblicitario creato con successo!", flush=True)
        return {"spot_url": public_video_url}

    except Exception as e:
        print(f"❌ Errore durante la creazione dello spot: {e}", flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
