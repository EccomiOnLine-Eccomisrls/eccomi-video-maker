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
# 2. CARICAMENTO MODELLO WAN 2.1 14B IMAGE-TO-VIDEO
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
    from diffusers.hooks import apply_group_offloading
    from diffusers.utils import export_to_video
    from transformers import (
        UMT5EncoderModel,
        CLIPVisionModel,
    )

    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    print(
        "--> [MODEL] Caricamento Wan 2.1 14B "
        "con SEQUENTIAL GROUP + DISK OFFLOADING...",
        flush=True,
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")

    # --------------------------------------------------------
    # DIRECTORY DI OFFLOAD SU DISCO
    # --------------------------------------------------------

    offload_root = "/tmp/wan_group_offload"

    text_offload_path = os.path.join(
        offload_root,
        "text_encoder",
    )

    transformer_offload_path = os.path.join(
        offload_root,
        "transformer",
    )

    os.makedirs(
        text_offload_path,
        exist_ok=True,
    )

    os.makedirs(
        transformer_offload_path,
        exist_ok=True,
    )

    print(
        f"--> [OFFLOAD] Directory principale: {offload_root}",
        flush=True,
    )

    # ========================================================
    # STEP 1 — IMAGE ENCODER
    # ========================================================

    print(
        "--> [MODEL 1/4] Carico CLIP Image Encoder...",
        flush=True,
    )

    image_encoder = CLIPVisionModel.from_pretrained(
        model_id,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )

    print(
        "--> [MODEL 1/4] CLIP Image Encoder caricato.",
        flush=True,
    )

    gc.collect()

    # ========================================================
    # STEP 2 — TEXT ENCODER
    # Carichiamo e OFFLOADIAMO SUBITO, prima del Transformer.
    # ========================================================

    print(
        "--> [MODEL 2/4] Carico UMT5 Text Encoder...",
        flush=True,
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )

    print(
        "--> [OFFLOAD 1/2] "
        "Configuro IMMEDIATAMENTE Text Encoder su disco...",
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

    print(
        "--> [OFFLOAD 1/2] Text Encoder configurato con successo.",
        flush=True,
    )

    # Liberazione memoria PRIMA di caricare VAE e Transformer
    torch.cuda.empty_cache()
    gc.collect()

    # ========================================================
    # STEP 3 — VAE
    # ========================================================

    print(
        "--> [MODEL 3/4] Carico Wan VAE...",
        flush=True,
    )

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    print(
        "--> [MODEL 3/4] Wan VAE caricato.",
        flush=True,
    )

    torch.cuda.empty_cache()
    gc.collect()

    # ========================================================
    # STEP 4 — TRANSFORMER 14B
    #
    # IMPORTANTE:
    # Il Text Encoder è già sotto Group/Disk Offload.
    # Non abbiamo più Text Encoder + Transformer 14B
    # entrambi completamente residenti in RAM prima
    # dell'applicazione dell'offload.
    # ========================================================

    print(
        "--> [MODEL 4/4] Carico Wan Transformer 14B...",
        flush=True,
    )

    transformer = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    print(
        "--> [OFFLOAD 2/2] "
        "Configuro IMMEDIATAMENTE Transformer su disco...",
        flush=True,
    )

    transformer.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
        use_stream=False,
        offload_to_disk_path=transformer_offload_path,
    )

    print(
        "--> [OFFLOAD 2/2] Transformer configurato con successo.",
        flush=True,
    )

    torch.cuda.empty_cache()
    gc.collect()

    # ========================================================
    # CREAZIONE PIPELINE
    # ========================================================

    print(
        "--> [MODEL] Creo Wan Image-To-Video Pipeline...",
        flush=True,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
    )

    # Manteniamo il comportamento della release che
    # è già riuscita a completare un video.
    #
    # Diffusers può mostrare un warning sui moduli già
    # group-offloaded: gli hook continueranno comunque
    # a gestire Text Encoder e Transformer.
    pipe.to("cuda")

    torch.cuda.empty_cache()
    gc.collect()

    print(
        "--> [MODEL] Wan 2.1 14B 720P + "
        "SEQUENTIAL GROUP/DISK OFFLOAD caricato con successo!",
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
# 3. DOWNLOAD IMMAGINE
# ============================================================

def download_image(url: str) -> Image.Image:
    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    return Image.open(
        BytesIO(response.content)
    ).convert("RGB")


# ============================================================
# 4. DOWNLOAD FILE
# ============================================================

def download_file(
    url: str,
    output_path: str,
):
    response = requests.get(
        url,
        timeout=60,
    )

    response.raise_for_status()

    with open(
        output_path,
        "wb",
    ) as f:
        f.write(response.content)


# ============================================================
# 5. AUDIO
# ============================================================

def extract_or_download_audio(
    url: str,
    output_audio_path: str,
):

    temp_download = tempfile.NamedTemporaryFile(
        delete=False
    ).name

    download_file(
        url,
        temp_download,
    )

    video_extensions = [
        ".mov",
        ".mp4",
        ".avi",
        ".mkv",
        ".webm",
        ".quicktime",
    ]

    is_video = any(
        url.lower().endswith(ext)
        for ext in video_extensions
    )

    if is_video:

        print(
            f"--> [AUDIO] Estraggo l'audio da file video: {url}...",
            flush=True,
        )

        video_clip = VideoFileClip(
            temp_download
        )

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

        os.rename(
            temp_download,
            output_audio_path,
        )


# ============================================================
# 6. GENERAZIONE SINGOLA CLIP WAN
# ============================================================

def generate_single_clip_wan(
    image: Image.Image,
    prompt: str,
) -> str:

    print(
        f"--> [WAN 2.1 ECCOMI] Genero scena: '{prompt}'",
        flush=True,
    )

    # Prompt automatico specifico per mascotte illustrate ECCOMI.
    # Manteniamo il preset già validato tecnicamente.
    enhanced_prompt = (
        f"{prompt}, "
        f"premium comic illustration, "
        f"branded mascot animation, "
        f"consistent character identity, "
        f"stable face, "
        f"stable body proportions, "
        f"stable costume details, "
        f"smooth subtle motion, "
        f"clean professional quality"
    )

    torch.cuda.empty_cache()
    gc.collect()

    print(
        "--> [WAN] Avvio inferenza 720P / 33 frames / 30 steps...",
        flush=True,
    )

    frames = pipe(
        image=image,
        prompt=enhanced_prompt,

        height=720,
        width=1280,

        num_frames=33,
        num_inference_steps=30,

        guidance_scale=5.0,
    ).frames[0]

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
        fps=24,
    )

    torch.cuda.empty_cache()
    gc.collect()

    return temp_file.name


# ============================================================
# 7. HANDLER RUNPOD
# ============================================================

def handler(event):

    print(
        "--- 🚀 Avvio Generazione Spot Pubblicitario "
        "ECCOMI con Wan 2.1 14B ---",
        flush=True,
    )

    job_input = event.get(
        "input",
        {},
    )

    job_id = event.get(
        "id",
        "test_job",
    )

    image_url = job_input.get(
        "image_url"
    )

    voice_url = job_input.get(
        "voice_audio_url"
    )

    music_url = job_input.get(
        "music_audio_url"
    )

    scenes_prompts = job_input.get(
        "scenes_prompts",
        [
            (
                "ECCOMI MAN standing confidently and presenting "
                "the ECCOMI ONLINE ecosystem with his open hand. "
                "Preserve exactly the official mascot."
            )
        ],
    )

    if not image_url:

        return {
            "error": "Nessuna image_url fornita."
        }

    generated_clip_paths = []
    temp_audio_files = []

    try:

        # ----------------------------------------------------
        # DOWNLOAD IMMAGINE
        # ----------------------------------------------------

        print(
            "--> [INPUT] Download immagine...",
            flush=True,
        )

        init_image = download_image(
            image_url
        )

        print(
            f"--> [INPUT] Immagine ricevuta: "
            f"{init_image.width}x{init_image.height}",
            flush=True,
        )

        # ----------------------------------------------------
        # GENERAZIONE SCENE
        # ----------------------------------------------------

        for idx, scene_prompt in enumerate(
            scenes_prompts
        ):

            print(
                f"--> Genero Scena "
                f"{idx + 1}/{len(scenes_prompts)} "
                f"con Wan 2.1...",
                flush=True,
            )

            clip_path = generate_single_clip_wan(
                init_image,
                scene_prompt,
            )

            generated_clip_paths.append(
                clip_path
            )

        # ----------------------------------------------------
        # MONTAGGIO
        # ----------------------------------------------------

        print(
            "--> [MONTAGGIO] "
            "Unione clip con dissolvenze incrociate...",
            flush=True,
        )

        video_clips = [
            VideoFileClip(p)
            for p in generated_clip_paths
        ]

        final_video_base = concatenate_videoclips(
            [
                clip.crossfadein(0.5)
                for clip in video_clips
            ],
            method="compose",
        )

        audio_tracks = []

        # ----------------------------------------------------
        # VOCE
        # ----------------------------------------------------

        if voice_url:

            voice_audio_path = (
                tempfile.NamedTemporaryFile(
                    suffix=".mp3",
                    delete=False,
                ).name
            )

            extract_or_download_audio(
                voice_url,
                voice_audio_path,
            )

            if (
                voice_audio_path
                and os.path.exists(
                    voice_audio_path
                )
            ):

                temp_audio_files.append(
                    voice_audio_path
                )

                voice_clip = AudioFileClip(
                    voice_audio_path
                )

                audio_tracks.append(
                    voice_clip
                )

        # ----------------------------------------------------
        # MUSICA
        # ----------------------------------------------------

        if music_url:

            music_audio_path = (
                tempfile.NamedTemporaryFile(
                    suffix=".mp3",
                    delete=False,
                ).name
            )

            extract_or_download_audio(
                music_url,
                music_audio_path,
            )

            if (
                music_audio_path
                and os.path.exists(
                    music_audio_path
                )
            ):

                temp_audio_files.append(
                    music_audio_path
                )

                music_clip = (
                    AudioFileClip(
                        music_audio_path
                    )
                    .volumex(0.12)
                    .set_duration(
                        final_video_base.duration
                    )
                )

                audio_tracks.append(
                    music_clip
                )

        # ----------------------------------------------------
        # MIX AUDIO
        # ----------------------------------------------------

        if audio_tracks:

            print(
                "--> [AUDIO] "
                "Unione tracce audio...",
                flush=True,
            )

            final_audio = CompositeAudioClip(
                audio_tracks
            )

            final_video = (
                final_video_base.set_audio(
                    final_audio
                )
            )

        else:

            final_video = final_video_base

        # ----------------------------------------------------
        # ESPORTAZIONE VIDEO
        # ----------------------------------------------------

        output_spot_path = (
            tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False,
            ).name
        )

        print(
            "--> [RENDER] Rendering finale...",
            flush=True,
        )

        final_video.write_videofile(
            output_spot_path,
            fps=24,
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
            "--> [UPLOAD] "
            "Caricamento spot su Supabase...",
            flush=True,
        )

        safe_job_id = str(
            job_id
        ).replace(
            "/",
            "_",
        )

        object_path = (
            f"{safe_job_id}_spot_wan21.mp4"
        )

        with open(
            output_spot_path,
            "rb",
        ) as f:

            supabase.storage.from_(
                "videos"
            ).upload(
                path=object_path,
                file=f,
                file_options={
                    "content-type": "video/mp4",
                    "upsert": "true",
                },
            )

        public_video_url = (
            f"{BASE_SUPABASE_URL}"
            f"/storage/v1/object/public/videos/"
            f"{object_path}"
        )

        # ----------------------------------------------------
        # PULIZIA
        # ----------------------------------------------------

        for clip in video_clips:
            clip.close()

        final_video.close()

        for p in (
            generated_clip_paths
            + temp_audio_files
            + [output_spot_path]
        ):

            if p and os.path.exists(p):
                os.remove(p)

        torch.cuda.empty_cache()
        gc.collect()

        print(
            "✅ Spot Wan2.1 creato con successo!",
            flush=True,
        )

        print(
            f"✅ URL VIDEO: {public_video_url}",
            flush=True,
        )

        return {
            "spot_url": public_video_url
        }

    except Exception as e:

        print(
            f"❌ Errore durante la creazione dello spot: {e}",
            flush=True,
        )

        torch.cuda.empty_cache()
        gc.collect()

        return {
            "error": str(e)
        }


# ============================================================
# 8. START RUNPOD SERVERLESS
# ============================================================

runpod.serverless.start(
    {
        "handler": handler
    }
)
