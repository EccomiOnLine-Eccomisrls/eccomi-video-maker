import gc
import os
import tempfile
from io import BytesIO
from PIL import Image
import requests
import runpod
from supabase import Client, create_client
import torch
import ftfy

# Import MoviePy v1.0.3
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    VideoFileClip,
    concatenate_videoclips,
)

print("--> [INIT] Avvio dello script handler.py (Wan 2.1 14B High-Quality)...", flush=True)

# 1. Variabili d'ambiente e Client Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BASE_SUPABASE_URL = SUPABASE_URL.rstrip("/")

if not BASE_SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ WARNING: SUPABASE_URL o SUPABASE_KEY non impostate!", flush=True)

supabase: Client = create_client(BASE_SUPABASE_URL, SUPABASE_KEY)

# 2. Caricamento del Modello Wan 2.1 14B HD Image-To-Video
print(
    "--> [MODEL] Caricamento di Wan 2.1 14B (720P) in corso su GPU PRO...",
    flush=True,
)

try:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    # Carica VAE in float32 e la pipeline in bfloat16
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16
    )

    # OTTIMIZZAZIONE VRAM FONDAMENTALE (Evita OOM su GPU 48GB)
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    print(
        "--> [MODEL] Wan 2.1 14B 720P Caricato ed Ottimizzato con successo!",
        flush=True,
    )

except Exception as e:
    print(
        f"⚠️ [MODEL FALLBACK] Impossibile caricare Wan 2.1 14B: {e}", flush=True
    )
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def download_file(url: str, output_path: str):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def extract_or_download_audio(url: str, output_audio_path: str):
    temp_download = tempfile.NamedTemporaryFile(delete=False).name
    download_file(url, temp_download)

    video_extensions = [".mov", ".mp4", ".avi", ".mkv", ".webm", ".quicktime"]
    is_video = any(url.lower().endswith(ext) for ext in video_extensions)

    if is_video:
        print(f"--> [AUDIO] Estraggo l'audio da file video: {url}...", flush=True)
        video_clip = VideoFileClip(temp_download)
        if video_clip.audio is not None:
            video_clip.audio.write_audiofile(output_audio_path, logger=None, fps=44100)
            video_clip.close()
        if os.path.exists(temp_download):
            os.remove(temp_download)
    else:
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        os.rename(temp_download, output_audio_path)


def generate_single_clip_wan(image: Image.Image, prompt: str) -> str:
    """Generazione cinematografica con Wan 2.1"""
    print(f"--> [WAN 2.1 CINEMATIC] Genero scena: '{prompt}'", flush=True)
    
    enhanced_prompt = (
        f"{prompt}, 8k resolution, photorealistic, cinematic lighting, "
        f"dynamic camera movement, 35mm film shot, smooth physics, masterpiece"
    )

    torch.cuda.empty_cache()
    gc.collect()

    frames = pipe(
        image=image,
        prompt=enhanced_prompt,
        num_frames=49,             # Riduce i frame da 81 a 49 (3x più veloce)
        num_inference_steps=30,    # Riduce i passi da 50 a 30 mantenendo un'ottima qualità
        guidance_scale=6.0,
    ).frames[0]


    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    export_to_video(frames, temp_file.name, fps=24)

    torch.cuda.empty_cache()
    gc.collect()

    return temp_file.name


def handler(event):
    print("--- 🚀 Avvio Generazione Spot Pubblicitario con Wan 2.1 14B ---", flush=True)
    job_input = event.get("input", {})
    job_id = event.get("id", "test_job")

    image_url = job_input.get("image_url")
    voice_url = job_input.get("voice_audio_url")
    music_url = job_input.get("music_audio_url")

    scenes_prompts = job_input.get(
        "scenes_prompts",
        [
            "Eccomi Man flying dynamically above futuristic night city, red cape fluttering in wind, cinematic lighting",
            "Eccomi Man landing heroically, pointing to a glowing interactive hologram screen showing energy savings",
            "Medium close up shot of Eccomi Man smiling confident, subtle camera rotation, cinematic office lighting",
            "Dramatic camera zoom out, Eccomi Man holding a superhero badge with glowing neon logo in background",
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
            print(f"--> Genero Scena {idx+1}/{len(scenes_prompts)} con Wan 2.1...", flush=True)
            clip_path = generate_single_clip_wan(init_image, scene_prompt)
            generated_clip_paths.append(clip_path)

        # 2. MONTAGGIO CON CROSSFADE
        print("--> [MONTAGGIO] Unione clip con dissolvenze incrociate...", flush=True)
        video_clips = [VideoFileClip(p) for p in generated_clip_paths]
        final_video_base = concatenate_videoclips(
            [clip.crossfadein(0.5) for clip in video_clips], 
            method="compose"
        )

        audio_tracks = []

        # 3. TRACCIA VOCALE
        if voice_url:
            voice_audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            extract_or_download_audio(voice_url, voice_audio_path)

            if voice_audio_path and os.path.exists(voice_audio_path):
                temp_audio_files.append(voice_audio_path)
                voice_clip = AudioFileClip(voice_audio_path)
                audio_tracks.append(voice_clip)

        # 4. MUSICA DI SOTTOFONDO
        if music_url:
            music_audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            extract_or_download_audio(music_url, music_audio_path)

            if music_audio_path and os.path.exists(music_audio_path):
                temp_audio_files.append(music_audio_path)
                music_clip = (
                    AudioFileClip(music_audio_path)
                    .volumex(0.12)
                    .set_duration(final_video_base.duration)
                )
                audio_tracks.append(music_clip)

        # Mix finale
        if audio_tracks:
            print("--> [AUDIO] Unione tracce audio...", flush=True)
            final_audio = CompositeAudioClip(audio_tracks)
            final_video = final_video_base.set_audio(final_audio)
        else:
            final_video = final_video_base

        # 5. ESPORTAZIONE
        output_spot_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        print("--> [RENDER] Rendering finale in corso...", flush=True)
        final_video.write_videofile(
            output_spot_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            bitrate="10000k",
        )

        # 6. UPLOAD SUPABASE
        print("--> [UPLOAD] Caricamento spot su Supabase...", flush=True)
        object_path = f"{job_id}_spot_wan21.mp4"

        with open(output_spot_path, "rb") as f:
            supabase.storage.from_("videos").upload(
                path=object_path,
                file=f,
                file_options={"content-type": "video/mp4", "upsert": "true"},
            )

        public_video_url = supabase.storage.from_("videos").get_public_url(object_path)

        # Pulizia
        for clip in video_clips:
            clip.close()
        final_video.close()

        for p in generated_clip_paths + temp_audio_files + [output_spot_path]:
            if p and os.path.exists(p):
                os.remove(p)

        print("✅ Spot Wan2.1 creato con successo!", flush=True)
        return {"spot_url": public_video_url}

    except Exception as e:
        print(f"❌ Errore durante la creazione dello spot: {e}", flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
