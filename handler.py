import gc
import os
import tempfile
from io import BytesIO

from PIL import Image, ImageOps
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

print(
    "--> [INIT] Avvio dello script handler.py (Wan 2.1 14B High-Quality)...",
    flush=True,
)

# ============================================================
# 1. VARIABILI D'AMBIENTE E CLIENT SUPABASE
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BASE_SUPABASE_URL = SUPABASE_URL.rstrip("/")

if not BASE_SUPABASE_URL or not SUPABASE_KEY:
    print(
        "⚠️ WARNING: SUPABASE_URL o SUPABASE_KEY non impostate!",
        flush=True,
    )

supabase: Client = create_client(
    BASE_SUPABASE_URL,
    SUPABASE_KEY,
)

# ============================================================
# 2. CONFIG GENERALE
# ============================================================

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Più sicuro su A40 rispetto a settaggi troppo aggressivi
WAN_NUM_FRAMES = 41
WAN_NUM_STEPS = 24
WAN_GUIDANCE_SCALE = 4.5

# FPS finale volutamente non troppo alto:
# con gli stessi frame ottieni una clip leggermente più lunga
EXPORT_FPS = 16

DEFAULT_SCENES = [
    (
        "ECCOMI MAN, same illustrated superhero mascot from the reference image, "
        "standing confidently, making a clear welcoming presentation gesture with one arm, "
        "slight torso movement, subtle head movement, natural facial expression change, "
        "gentle cape movement, stable character identity, stable costume details, "
        "clean blue futuristic background"
    ),
    (
        "ECCOMI MAN, same illustrated superhero mascot from the reference image, "
        "presenting the ECCOMI ONLINE world with an open hand, "
        "visible arm gesture, slight body shift, subtle smile change, "
        "small expressive movement of shoulders and head, "
        "soft cape flow, stable face and stable suit, "
        "clean professional branded animation"
    ),
]

# ============================================================
# 3. CARICAMENTO MODELLO WAN 2.1 14B IMAGE-TO-VIDEO
# ============================================================

print(
    "--> [MODEL] Caricamento di Wan 2.1 14B (720P) in corso...",
    flush=True,
)

try:
    from diffusers import (
        AutoencoderKLWan,
        WanTransformer3DModel,
        WanImageToVideoPipeline,
    )
    from diffusers.hooks.group_offloading import apply_group_offloading
    from diffusers.utils import export_to_video
    from transformers import (
        UMT5EncoderModel,
        CLIPVisionModel,
    )

    print(
        "--> [MODEL] Caricamento componenti Wan 2.1 14B con Group Offloading...",
        flush=True,
    )

    # --------------------------------------------------------
    # DIRECTORY OFFLOAD
    # --------------------------------------------------------

    offload_root = "/tmp/wan_group_offload"
    text_offload_path = os.path.join(offload_root, "text_encoder")
    transformer_offload_path = os.path.join(offload_root, "transformer")

    os.makedirs(text_offload_path, exist_ok=True)
    os.makedirs(transformer_offload_path, exist_ok=True)

    print(
        f"--> [OFFLOAD] Directory principale: {offload_root}",
        flush=True,
    )

    # --------------------------------------------------------
    # IMAGE ENCODER
    # --------------------------------------------------------

    print(
        "--> [MODEL] Carico CLIP Image Encoder...",
        flush=True,
    )

    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_ID,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )

    # --------------------------------------------------------
    # TEXT ENCODER
    # --------------------------------------------------------

    print(
        "--> [MODEL] Carico UMT5 Text Encoder...",
        flush=True,
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )

    # --------------------------------------------------------
    # VAE
    # --------------------------------------------------------

    print(
        "--> [MODEL] Carico Wan VAE...",
        flush=True,
    )

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    # --------------------------------------------------------
    # TRANSFORMER
    # --------------------------------------------------------

    print(
        "--> [MODEL] Carico Wan Transformer 14B...",
        flush=True,
    )

    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")

    # --------------------------------------------------------
    # OFFLOAD TEXT ENCODER
    # --------------------------------------------------------

    print(
        "--> [OFFLOAD] Configuro Text Encoder...",
        flush=True,
    )

    apply_group_offloading(
        text_encoder,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="block_level",
        num_blocks_per_group=4,
        use_stream=False,
        offload_to_disk_path=text_offload_path,
    )

    # --------------------------------------------------------
    # OFFLOAD TRANSFORMER
    # --------------------------------------------------------

    print(
        "--> [OFFLOAD] Configuro Transformer...",
        flush=True,
    )

    transformer.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
        use_stream=False,
        offload_to_disk_path=transformer_offload_path,
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    print(
        "--> [MODEL] Creo Wan Image-To-Video Pipeline...",
        flush=True,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
    )

    # Ottimizzazioni leggere e sicure
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    # Importante: lasciamo fare agli hook di offload
    pipe.to("cuda")

    torch.cuda.empty_cache()
    gc.collect()

    print(
        "--> [MODEL] Wan 2.1 14B 720P + GROUP OFFLOAD caricato con successo!",
        flush=True,
    )

except Exception as e:
    print(
        f"⚠️ [MODEL FALLBACK] Impossibile caricare Wan 2.1 14B: {e}",
        flush=True,
    )

    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()

# ============================================================
# 4. UTILS IMMAGINE
# ============================================================

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    return ftfy.fix_text(str(text)).strip()


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def prepare_image_for_wan(image: Image.Image) -> Image.Image:
    """
    Adatta l'immagine al formato 16:9 richiesto dal video finale,
    evitando risultati strani o bande non volute.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Crop intelligente centrato a 1280x720
    prepared = ImageOps.fit(
        image,
        (TARGET_WIDTH, TARGET_HEIGHT),
        method=Image.LANCZOS,
        centering=(0.5, 0.5),
    )

    return prepared

# ============================================================
# 5. DOWNLOAD FILE
# ============================================================

def download_file(url: str, output_path: str):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)

# ============================================================
# 6. AUDIO
# ============================================================

def extract_or_download_audio(url: str, output_audio_path: str):
    temp_download = tempfile.NamedTemporaryFile(delete=False).name

    download_file(url, temp_download)

    video_extensions = [
        ".mov",
        ".mp4",
        ".avi",
        ".mkv",
        ".webm",
        ".quicktime",
    ]

    is_video = any(url.lower().endswith(ext) for ext in video_extensions)

    if is_video:
        print(
            f"--> [AUDIO] Estraggo l'audio da file video: {url}...",
            flush=True,
        )

        video_clip = VideoFileClip(temp_download)

        if video_clip.audio is not None:
            video_clip.audio.write_audiofile(
                output_audio_path,
                logger=None,
                fps=44100,
            )

        video_clip.close()

        if os.path.exists(temp_download):
            os.remove(temp_download)

    else:
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)

        os.rename(temp_download, output_audio_path)

# ============================================================
# 7. COSTRUZIONE PROMPT
# ============================================================

def build_enhanced_prompt(prompt: str) -> str:
    """
    Prompt rinforzato:
    - stessa identità del personaggio
    - ma con movimento leggibile
    - senza congelarlo troppo
    """
    prompt = sanitize_text(prompt)

    base_suffix = (
        " premium comic illustration, branded mascot animation, "
        "same hero identity as the input image, same costume, same emblem, "
        "stable face, stable hairstyle, stable body proportions, "
        "visible but controlled arm motion, slight torso shift, "
        "small natural head movement, subtle expression change, "
        "gentle cape flow, clean professional quality, no extra arms, "
        "no extra hands, no body distortion, no costume redesign"
    )

    return f"{prompt}, {base_suffix}"

# ============================================================
# 8. GENERAZIONE SINGOLA CLIP WAN
# ============================================================

def generate_single_clip_wan(
    image: Image.Image,
    prompt: str,
) -> str:

    print(
        f"--> [WAN 2.1 ECCOMI] Genero scena: '{prompt}'",
        flush=True,
    )

    enhanced_prompt = build_enhanced_prompt(prompt)

    torch.cuda.empty_cache()
    gc.collect()

    print(
        f"--> [WAN] Avvio inferenza 720P / {WAN_NUM_FRAMES} frames / {WAN_NUM_STEPS} steps...",
        flush=True,
    )

    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=enhanced_prompt,
            height=TARGET_HEIGHT,
            width=TARGET_WIDTH,
            num_frames=WAN_NUM_FRAMES,
            num_inference_steps=WAN_NUM_STEPS,
            guidance_scale=WAN_GUIDANCE_SCALE,
        )

    frames = result.frames[0]

    print(
        "--> [WAN] Inferenza completata.",
        flush=True,
    )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    )

    export_to_video(
        frames,
        temp_file.name,
        fps=EXPORT_FPS,
    )

    torch.cuda.empty_cache()
    gc.collect()

    return temp_file.name

# ============================================================
# 9. HANDLER RUNPOD
# ============================================================

def handler(event):

    print(
        "--- 🚀 Avvio Generazione Spot Pubblicitario ECCOMI con Wan 2.1 14B ---",
        flush=True,
    )

    job_input = event.get("input", {})
    job_id = event.get("id", "test_job")

    image_url = job_input.get("image_url")
    voice_url = job_input.get("voice_audio_url")
    music_url = job_input.get("music_audio_url")

    scenes_prompts = job_input.get("scenes_prompts", DEFAULT_SCENES)

    if not image_url:
        return {"error": "Nessuna image_url fornita."}

    generated_clip_paths = []
    temp_audio_files = []
    video_clips = []
    final_video = None
    final_video_base = None

    try:
        # ----------------------------------------------------
        # DOWNLOAD IMMAGINE
        # ----------------------------------------------------

        print(
            "--> [INPUT] Download immagine...",
            flush=True,
        )

        init_image = download_image(image_url)

        print(
            f"--> [INPUT] Immagine originale ricevuta: {init_image.width}x{init_image.height}",
            flush=True,
        )

        init_image = prepare_image_for_wan(init_image)

        print(
            f"--> [INPUT] Immagine adattata per Wan: {init_image.width}x{init_image.height}",
            flush=True,
        )

        # ----------------------------------------------------
        # GENERAZIONE SCENE
        # ----------------------------------------------------

        if not isinstance(scenes_prompts, list) or len(scenes_prompts) == 0:
            scenes_prompts = DEFAULT_SCENES

        for idx, scene_prompt in enumerate(scenes_prompts):
            scene_prompt = sanitize_text(scene_prompt)

            if not scene_prompt:
                continue

            print(
                f"--> Genero Scena {idx + 1}/{len(scenes_prompts)} con Wan 2.1...",
                flush=True,
            )

            clip_path = generate_single_clip_wan(
                init_image,
                scene_prompt,
            )

            generated_clip_paths.append(clip_path)

        if len(generated_clip_paths) == 0:
            return {"error": "Nessuna clip generata."}

        # ----------------------------------------------------
        # MONTAGGIO
        # ----------------------------------------------------

        print(
            "--> [MONTAGGIO] Unione clip...",
            flush=True,
        )

        video_clips = [VideoFileClip(p) for p in generated_clip_paths]

        if len(video_clips) == 1:
            final_video_base = video_clips[0]
        else:
            final_video_base = concatenate_videoclips(
                [clip.crossfadein(0.25) for clip in video_clips],
                method="compose",
            )

        audio_tracks = []

        # ----------------------------------------------------
        # VOCE
        # ----------------------------------------------------

        if voice_url:
            voice_audio_path = tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False,
            ).name

            extract_or_download_audio(
                voice_url,
                voice_audio_path,
            )

            if voice_audio_path and os.path.exists(voice_audio_path):
                temp_audio_files.append(voice_audio_path)
                voice_clip = AudioFileClip(voice_audio_path)
                audio_tracks.append(voice_clip)

        # ----------------------------------------------------
        # MUSICA
        # ----------------------------------------------------

        if music_url:
            music_audio_path = tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False,
            ).name

            extract_or_download_audio(
                music_url,
                music_audio_path,
            )

            if music_audio_path and os.path.exists(music_audio_path):
                temp_audio_files.append(music_audio_path)

                music_clip = (
                    AudioFileClip(music_audio_path)
                    .volumex(0.12)
                    .set_duration(final_video_base.duration)
                )

                audio_tracks.append(music_clip)

        # ----------------------------------------------------
        # MIX AUDIO
        # ----------------------------------------------------

        if audio_tracks:
            print(
                "--> [AUDIO] Unione tracce audio...",
                flush=True,
            )

            final_audio = CompositeAudioClip(audio_tracks)
            final_video = final_video_base.set_audio(final_audio)
        else:
            final_video = final_video_base

        # ----------------------------------------------------
        # ESPORTAZIONE VIDEO
        # ----------------------------------------------------

        output_spot_path = tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ).name

        print(
            "--> [RENDER] Rendering finale...",
            flush=True,
        )

        final_video.write_videofile(
            output_spot_path,
            fps=EXPORT_FPS,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            bitrate="10000k",
            ffmpeg_params=[
                "-pix_fmt",
                "yuv420p",
            ],
        )

        # ----------------------------------------------------
        # UPLOAD SUPABASE
        # ----------------------------------------------------

        print(
            "--> [UPLOAD] Caricamento spot su Supabase...",
            flush=True,
        )

        safe_job_id = str(job_id).replace("/", "_")
        object_path = f"{safe_job_id}_spot_wan21.mp4"

        with open(output_spot_path, "rb") as f:
            supabase.storage.from_("videos").upload(
                path=object_path,
                file=f,
                file_options={
                    "content-type": "video/mp4",
                    "upsert": "true",
                },
            )

        public_video_url = (
            f"{BASE_SUPABASE_URL}/storage/v1/object/public/videos/{object_path}"
        )

        print(
            f"✅ URL VIDEO: {public_video_url}",
            flush=True,
        )

        return {"spot_url": public_video_url}

    except Exception as e:
        print(
            f"❌ Errore durante la creazione dello spot: {e}",
            flush=True,
        )

        torch.cuda.empty_cache()
        gc.collect()

        return {"error": str(e)}

    finally:
        # ----------------------------------------------------
        # PULIZIA
        # ----------------------------------------------------
        try:
            for clip in video_clips:
                try:
                    clip.close()
                except Exception:
                    pass

            if final_video and final_video is not final_video_base:
                try:
                    final_video.close()
                except Exception:
                    pass

            if final_video_base:
                try:
                    final_video_base.close()
                except Exception:
                    pass

            for p in generated_clip_paths + temp_audio_files:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        except Exception:
            pass

        torch.cuda.empty_cache()
        gc.collect()

# ============================================================
# 10. START RUNPOD SERVERLESS
# ============================================================

runpod.serverless.start(
    {
        "handler": handler
    }
)
