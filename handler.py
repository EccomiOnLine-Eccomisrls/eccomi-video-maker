import gc
import os
import tempfile
import time
from io import BytesIO

from PIL import Image
import requests
import runpod
from supabase import Client, create_client
import torch
import ftfy

# MoviePy v1.0.3
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    VideoFileClip,
    concatenate_videoclips,
)


# ============================================================
# 0. CONFIG PYTORCH / CUDA
# ============================================================

print(
    "--> [INIT] Avvio handler.py "
    "(Wan 2.1 14B - 80GB FULL GPU MODE + AUTO DELIVERY)...",
    flush=True,
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ============================================================
# 1. VARIABILI D'AMBIENTE E CLIENT SUPABASE
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BASE_SUPABASE_URL = SUPABASE_URL.rstrip("/")

# ------------------------------------------------------------
# AUTO DELIVERY
#
# EVS_CALLBACK_URL:
# URL opzionale a cui inviare automaticamente il risultato
# quando il video Ã¨ pronto.
#
# PuÃ² anche essere passato per singolo job tramite:
# input.callback_url
#
# EVS_CALLBACK_TOKEN:
# token opzionale salvato come secret/env RunPod.
# Se presente viene inviato come:
# Authorization: Bearer <token>
# ------------------------------------------------------------

DEFAULT_EVS_CALLBACK_URL = (
    f"{BASE_SUPABASE_URL}/functions/v1/evs-video-callback"
    if BASE_SUPABASE_URL
    else ""
)

EVS_CALLBACK_URL = os.environ.get(
    "EVS_CALLBACK_URL",
    DEFAULT_EVS_CALLBACK_URL,
).strip()

EVS_CALLBACK_TOKEN = os.environ.get(
    "EVS_CALLBACK_TOKEN",
    "",
).strip()

CALLBACK_TIMEOUT_SECONDS = 20
CALLBACK_MAX_ATTEMPTS = 3


if not BASE_SUPABASE_URL or not SUPABASE_KEY:
    print(
        "â ï¸ WARNING: SUPABASE_URL o SUPABASE_KEY non impostate!",
        flush=True,
    )

supabase: Client = create_client(
    BASE_SUPABASE_URL,
    SUPABASE_KEY,
)


# ============================================================
# 2. CONFIG WAN
# ============================================================

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

WAN_NUM_FRAMES = 49
WAN_NUM_STEPS = 30
WAN_GUIDANCE_SCALE = 5.0
EXPORT_FPS = 24


DEFAULT_SCENES = [
    (
        "ECCOMI MAN, the exact official illustrated superhero mascot "
        "from the reference image. "
        "ONLY ONE MOTION IS ALLOWED. "
        "The arm already extended on the RIGHT SIDE OF THE IMAGE, "
        "viewer's right, slowly and continuously extends slightly farther "
        "outward and slightly upward from the first frame until the final frame. "
        "The motion must be clearly visible but elegant and controlled. "
        "Do not return the presenting arm to its starting position. "
        "The hand resting on the hip on the LEFT SIDE OF THE IMAGE, "
        "viewer's left, must remain completely frozen and attached to the hip "
        "from first frame to last frame. "
        "The head, face, shoulders, torso, waist, legs, feet and red cape "
        "must remain completely still. "
        "Fixed camera. No zoom. No camera movement. No walking. "
        "No second-arm movement. No pose change. No cape movement. "
        "No head movement. No torso movement. "
        "Preserve exactly his face, hairstyle, body proportions, costume, "
        "chest emblem, gloves, boots and cape. "
        "No morphing. No extra fingers. No extra hands. No extra limbs."
    )
]


# ============================================================
# 3. LOG MEMORIA GPU
# ============================================================

def log_gpu_memory(stage: str):
    if not torch.cuda.is_available():
        print(
            f"--> [GPU] {stage}: CUDA non disponibile.",
            flush=True,
        )
        return

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)

    total_gb = props.total_memory / (1024 ** 3)

    allocated_gb = (
        torch.cuda.memory_allocated(device_index)
        / (1024 ** 3)
    )

    reserved_gb = (
        torch.cuda.memory_reserved(device_index)
        / (1024 ** 3)
    )

    max_allocated_gb = (
        torch.cuda.max_memory_allocated(device_index)
        / (1024 ** 3)
    )

    print(
        f"--> [GPU] {stage} | "
        f"GPU={props.name} | "
        f"VRAM totale={total_gb:.2f} GiB | "
        f"allocata={allocated_gb:.2f} GiB | "
        f"riservata={reserved_gb:.2f} GiB | "
        f"picco={max_allocated_gb:.2f} GiB",
        flush=True,
    )


# ============================================================
# 4. CARICAMENTO MODELLO
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA non disponibile. "
        "Questo handler richiede una GPU NVIDIA."
    )


print(
    "--> [MODEL] ModalitÃ  FULL GPU 80GB.",
    flush=True,
)

print(
    "--> [MODEL] NESSUN CPU OFFLOAD.",
    flush=True,
)

print(
    "--> [MODEL] NESSUN DISK OFFLOAD.",
    flush=True,
)

print(
    f"--> [MODEL] Caricamento {MODEL_ID} direttamente su CUDA...",
    flush=True,
)


model_load_start = time.perf_counter()


try:
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    log_gpu_memory("Prima del caricamento modello")

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )

    if hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
            print(
                "--> [MODEL] VAE slicing attivo.",
                flush=True,
            )
        except Exception as e:
            print(
                f"--> [MODEL] VAE slicing non applicato: {e}",
                flush=True,
            )

    if hasattr(pipe, "enable_vae_tiling"):
        try:
            pipe.enable_vae_tiling()
            print(
                "--> [MODEL] VAE tiling attivo.",
                flush=True,
            )
        except Exception as e:
            print(
                f"--> [MODEL] VAE tiling non applicato: {e}",
                flush=True,
            )

    pipe.to("cuda")

    torch.cuda.empty_cache()
    gc.collect()

    model_load_seconds = (
        time.perf_counter() - model_load_start
    )

    print(
        f"â [MODEL] Wan 2.1 14B FULL GPU caricato "
        f"in {model_load_seconds:.1f} secondi.",
        flush=True,
    )

    log_gpu_memory("Dopo caricamento modello")

except Exception as e:
    print(
        "â [MODEL] ERRORE caricamento Wan FULL GPU:",
        flush=True,
    )

    print(
        str(e),
        flush=True,
    )

    raise


# ============================================================
# 5. UTILS
# ============================================================

def sanitize_text(text: str) -> str:
    if not text:
        return ""

    return ftfy.fix_text(
        str(text)
    ).strip()


def make_temp_path(suffix: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    )

    path = temp_file.name
    temp_file.close()

    return path


def sanitize_job_id(value) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
    )


# ============================================================
# 6. AUTO DELIVERY / CALLBACK
# ============================================================

def send_delivery_callback(
    callback_url: str,
    payload: dict,
) -> dict:

    callback_url = sanitize_text(
        callback_url
    )

    if not callback_url:
        print(
            "--> [DELIVERY] Nessun callback_url configurato. "
            "Il risultato resta disponibile nell'output RunPod.",
            flush=True,
        )

        return {
            "enabled": False,
            "delivered": False,
            "status": "SKIPPED",
        }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "EVS-PRO-RunPod/1.0",
    }

    callback_token = (
        EVS_CALLBACK_TOKEN
        or SUPABASE_KEY
    )

    if callback_token:
        headers["Authorization"] = (
            f"Bearer {callback_token}"
        )

    if SUPABASE_KEY:
        headers["apikey"] = SUPABASE_KEY

    last_error = None

    print(
        f"--> [DELIVERY] Invio callback a: {callback_url}",
        flush=True,
    )

    for attempt in range(
        1,
        CALLBACK_MAX_ATTEMPTS + 1,
    ):

        try:
            response = requests.post(
                callback_url,
                json=payload,
                headers=headers,
                timeout=CALLBACK_TIMEOUT_SECONDS,
            )

            if 200 <= response.status_code < 300:
                print(
                    f"â [DELIVERY] Callback consegnato "
                    f"(HTTP {response.status_code}) "
                    f"tentativo {attempt}/{CALLBACK_MAX_ATTEMPTS}.",
                    flush=True,
                )

                return {
                    "enabled": True,
                    "delivered": True,
                    "status": "DELIVERED",
                    "http_status": response.status_code,
                    "attempts": attempt,
                }

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

            print(
                f"â ï¸ [DELIVERY] Tentativo {attempt} fallito: "
                f"{last_error}",
                flush=True,
            )

        except Exception as e:
            last_error = str(e)

            print(
                f"â ï¸ [DELIVERY] Tentativo {attempt} fallito: "
                f"{last_error}",
                flush=True,
            )

        if attempt < CALLBACK_MAX_ATTEMPTS:
            time.sleep(
                min(
                    2 ** (attempt - 1),
                    4,
                )
            )

    print(
        f"â [DELIVERY] Callback non consegnato dopo "
        f"{CALLBACK_MAX_ATTEMPTS} tentativi. "
        f"Errore: {last_error}",
        flush=True,
    )

    return {
        "enabled": True,
        "delivered": False,
        "status": "FAILED",
        "attempts": CALLBACK_MAX_ATTEMPTS,
        "error": last_error,
    }


# ============================================================
# 7. DOWNLOAD IMMAGINE
# ============================================================

def download_image(url: str) -> Image.Image:
    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    return image


# ============================================================
# 8. DOWNLOAD FILE
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
# 9. AUDIO
# ============================================================

def extract_or_download_audio(
    url: str,
    output_audio_path: str,
):
    temp_download = make_temp_path("")

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

    clean_url = (
        url.lower()
        .split("?")[0]
        .split("#")[0]
    )

    is_video = any(
        clean_url.endswith(ext)
        for ext in video_extensions
    )

    if is_video:

        print(
            f"--> [AUDIO] Estraggo audio da video: {url}",
            flush=True,
        )

        video_clip = None

        try:
            video_clip = VideoFileClip(
                temp_download
            )

            if video_clip.audio is not None:
                video_clip.audio.write_audiofile(
                    output_audio_path,
                    logger=None,
                    fps=44100,
                )

        finally:
            if video_clip is not None:
                try:
                    video_clip.close()
                except Exception:
                    pass

            if os.path.exists(temp_download):
                try:
                    os.remove(temp_download)
                except Exception:
                    pass

    else:

        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)

        os.rename(
            temp_download,
            output_audio_path,
        )


# ============================================================
# 10. COSTRUZIONE PROMPT ECCOMI
# ============================================================

def build_enhanced_prompt(
    prompt: str,
) -> str:

    prompt = sanitize_text(prompt)

    base_suffix = (
        "premium comic illustration, "
        "professional branded mascot animation, "
        "same character identity as the input image, "
        "stable recognizable face, "
        "stable hairstyle, "
        "stable muscular body proportions, "
        "stable costume and chest emblem, "
        "smooth temporal consistency, "
        "clean professional quality, "
        "no character redesign, "
        "no face morphing, "
        "no costume morphing, "
        "no extra arms, "
        "no extra hands, "
        "no extra fingers, "
        "no extra legs, "
        "no body distortion"
    )

    return (
        f"{prompt}, "
        f"{base_suffix}"
    )


# ============================================================
# 11. GENERAZIONE SINGOLA CLIP WAN
# ============================================================

def generate_single_clip_wan(
    image: Image.Image,
    prompt: str,
) -> str:

    print(
        "--> [WAN 2.1 ECCOMI] Generazione nuova scena...",
        flush=True,
    )

    print(
        f"--> [PROMPT] {prompt}",
        flush=True,
    )

    enhanced_prompt = build_enhanced_prompt(
        prompt
    )

    print(
        f"--> [PROMPT FINALE WAN] {enhanced_prompt}",
        flush=True,
    )

    gc.collect()
    torch.cuda.empty_cache()

    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    log_gpu_memory(
        "Prima inferenza"
    )

    print(
        f"--> [WAN] Avvio inferenza "
        f"{TARGET_WIDTH}x{TARGET_HEIGHT} / "
        f"{WAN_NUM_FRAMES} frames / "
        f"{WAN_NUM_STEPS} steps / "
        f"guidance {WAN_GUIDANCE_SCALE}...",
        flush=True,
    )

    inference_start = time.perf_counter()

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

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    frames = result.frames[0]

    print(
        f"â [WAN] Inferenza completata in "
        f"{inference_seconds:.1f} secondi "
        f"({inference_seconds / 60:.2f} minuti).",
        flush=True,
    )

    log_gpu_memory(
        "Dopo inferenza"
    )

    temp_video_path = make_temp_path(
        ".mp4"
    )

    print(
        f"--> [WAN] Export clip a {EXPORT_FPS} fps...",
        flush=True,
    )

    export_to_video(
        frames,
        temp_video_path,
        fps=EXPORT_FPS,
    )

    del frames
    del result

    gc.collect()
    torch.cuda.empty_cache()

    return temp_video_path


# ============================================================
# 12. HANDLER RUNPOD
# ============================================================

def handler(event):

    job_start = time.perf_counter()

    print(
        "--- ð ECCOMI VIDEO MAKER "
        "| WAN 2.1 14B "
        "| FULL GPU 80GB "
        "| AUTO DELIVERY ---",
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
        DEFAULT_SCENES,
    )

    # --------------------------------------------------------
    # AUTO DELIVERY:
    # prioritÃ  a callback_url del singolo job;
    # altrimenti usa EVS_CALLBACK_URL configurato come env.
    # --------------------------------------------------------

    callback_url = sanitize_text(
        job_input.get(
            "callback_url"
        )
        or EVS_CALLBACK_URL
    )

    # Riferimento opzionale dell'ordine/cliente.
    # Utile per associare il callback al record corretto
    # nel gestionale EVS PRO.
    customer_reference = sanitize_text(
        job_input.get(
            "customer_reference"
        )
    )

    if not image_url:

        error_payload = {
            "event": "evs.video.failed",
            "status": "FAILED",
            "job_id": str(job_id),
            "customer_reference": customer_reference or None,
            "error": "Nessuna image_url fornita.",
        }

        delivery = send_delivery_callback(
            callback_url,
            error_payload,
        )

        return {
            **error_payload,
            "delivery": delivery,
        }

    if (
        not isinstance(scenes_prompts, list)
        or len(scenes_prompts) == 0
    ):
        scenes_prompts = DEFAULT_SCENES

    generated_clip_paths = []
    temp_audio_files = []
    video_clips = []
    audio_clips = []

    output_spot_path = None
    final_video_base = None
    final_video = None
    final_audio = None

    try:

        # ====================================================
        # DOWNLOAD IMMAGINE
        # ====================================================

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

        # ====================================================
        # GENERAZIONE SCENE
        # ====================================================

        valid_prompts = []

        for scene_prompt in scenes_prompts:
            clean_prompt = sanitize_text(
                scene_prompt
            )

            if clean_prompt:
                valid_prompts.append(
                    clean_prompt
                )

        if not valid_prompts:
            valid_prompts = DEFAULT_SCENES

        for idx, scene_prompt in enumerate(
            valid_prompts
        ):

            print(
                f"--> [SCENA] "
                f"{idx + 1}/{len(valid_prompts)}",
                flush=True,
            )

            clip_path = generate_single_clip_wan(
                init_image,
                scene_prompt,
            )

            generated_clip_paths.append(
                clip_path
            )

        if not generated_clip_paths:
            raise RuntimeError(
                "Nessuna clip generata."
            )

        # ====================================================
        # MONTAGGIO VIDEO
        # ====================================================

        print(
            "--> [MONTAGGIO] Apro clip generate...",
            flush=True,
        )

        for path in generated_clip_paths:
            video_clips.append(
                VideoFileClip(path)
            )

        if len(video_clips) == 1:

            final_video_base = video_clips[0]

        else:

            print(
                "--> [MONTAGGIO] "
                "Unione clip con crossfade...",
                flush=True,
            )

            final_video_base = concatenate_videoclips(
                [
                    clip.crossfadein(0.25)
                    for clip in video_clips
                ],
                method="compose",
            )

        # ====================================================
        # AUDIO VOCE
        # ====================================================

        if voice_url:

            voice_audio_path = make_temp_path(
                ".mp3"
            )

            extract_or_download_audio(
                voice_url,
                voice_audio_path,
            )

            if os.path.exists(
                voice_audio_path
            ):

                temp_audio_files.append(
                    voice_audio_path
                )

                voice_clip = AudioFileClip(
                    voice_audio_path
                )

                audio_clips.append(
                    voice_clip
                )

        # ====================================================
        # AUDIO MUSICA
        # ====================================================

        if music_url:

            music_audio_path = make_temp_path(
                ".mp3"
            )

            extract_or_download_audio(
                music_url,
                music_audio_path,
            )

            if os.path.exists(
                music_audio_path
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

                audio_clips.append(
                    music_clip
                )

        # ====================================================
        # MIX AUDIO
        # ====================================================

        if audio_clips:

            print(
                "--> [AUDIO] Mix tracce...",
                flush=True,
            )

            final_audio = CompositeAudioClip(
                audio_clips
            )

            final_video = (
                final_video_base
                .set_audio(final_audio)
            )

        else:

            final_video = final_video_base

        # ====================================================
        # RENDER FINALE
        # ====================================================

        output_spot_path = make_temp_path(
            ".mp4"
        )

        print(
            "--> [RENDER] Rendering finale...",
            flush=True,
        )

        render_start = time.perf_counter()

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

        render_seconds = (
            time.perf_counter()
            - render_start
        )

        print(
            f"â [RENDER] Completato in "
            f"{render_seconds:.1f} secondi.",
            flush=True,
        )

        # ====================================================
        # UPLOAD SUPABASE
        # ====================================================

        print(
            "--> [UPLOAD] Caricamento su Supabase...",
            flush=True,
        )

        safe_job_id = sanitize_job_id(
            job_id
        )

        object_path = (
            f"{safe_job_id}_spot_wan21.mp4"
        )

        with open(
            output_spot_path,
            "rb",
        ) as file_handle:

            supabase.storage.from_(
                "videos"
            ).upload(
                path=object_path,
                file=file_handle,
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

        total_job_seconds = (
            time.perf_counter()
            - job_start
        )

        generation_info = {
            "model": MODEL_ID,
            "mode": "full_gpu_80gb",
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "frames": WAN_NUM_FRAMES,
            "steps": WAN_NUM_STEPS,
            "guidance_scale": WAN_GUIDANCE_SCALE,
            "fps": EXPORT_FPS,
            "scene_count": len(valid_prompts),
            "total_seconds": round(
                total_job_seconds,
                2,
            ),
        }

        print(
            "============================================",
            flush=True,
        )

        print(
            "â ECCOMI VIDEO COMPLETATO",
            flush=True,
        )

        print(
            f"â TEMPO TOTALE JOB: "
            f"{total_job_seconds:.1f} sec "
            f"({total_job_seconds / 60:.2f} min)",
            flush=True,
        )

        print(
            f"â URL VIDEO: {public_video_url}",
            flush=True,
        )

        print(
            "============================================",
            flush=True,
        )

        # ====================================================
        # AUTO DELIVERY DEL LINK
        # ====================================================

        callback_payload = {
            "event": "evs.video.completed",
            "status": "COMPLETED",
            "job_id": str(job_id),
            "customer_reference": customer_reference or None,
            "spot_url": public_video_url,
            "generation": generation_info,
        }

        delivery = send_delivery_callback(
            callback_url,
            callback_payload,
        )

        return {
            "status": "COMPLETED",
            "job_id": str(job_id),
            "customer_reference": customer_reference or None,
            "spot_url": public_video_url,
            "generation": generation_info,
            "delivery": delivery,
        }

    except torch.cuda.OutOfMemoryError as e:

        print(
            "â [CUDA OOM] Memoria GPU insufficiente.",
            flush=True,
        )

        print(
            str(e),
            flush=True,
        )

        log_gpu_memory(
            "CUDA OOM"
        )

        gc.collect()
        torch.cuda.empty_cache()

        error_payload = {
            "event": "evs.video.failed",
            "status": "FAILED",
            "job_id": str(job_id),
            "customer_reference": customer_reference or None,
            "error": "CUDA_OUT_OF_MEMORY",
            "details": str(e),
        }

        delivery = send_delivery_callback(
            callback_url,
            error_payload,
        )

        return {
            **error_payload,
            "delivery": delivery,
        }

    except Exception as e:

        print(
            f"â [ERROR] {type(e).__name__}: {e}",
            flush=True,
        )

        log_gpu_memory(
            "Errore job"
        )

        gc.collect()
        torch.cuda.empty_cache()

        error_payload = {
            "event": "evs.video.failed",
            "status": "FAILED",
            "job_id": str(job_id),
            "customer_reference": customer_reference or None,
            "error": str(e),
        }

        delivery = send_delivery_callback(
            callback_url,
            error_payload,
        )

        return {
            **error_payload,
            "delivery": delivery,
        }

    finally:

        # ====================================================
        # CLEANUP MOVIEPY
        # ====================================================

        try:

            if final_audio is not None:
                try:
                    final_audio.close()
                except Exception:
                    pass

            for audio_clip in audio_clips:
                try:
                    audio_clip.close()
                except Exception:
                    pass

            if (
                final_video is not None
                and final_video is not final_video_base
            ):
                try:
                    final_video.close()
                except Exception:
                    pass

            if (
                final_video_base is not None
                and final_video_base not in video_clips
            ):
                try:
                    final_video_base.close()
                except Exception:
                    pass

            for video_clip in video_clips:
                try:
                    video_clip.close()
                except Exception:
                    pass

        except Exception:
            pass

        # ====================================================
        # CLEANUP FILE TEMPORANEI
        # ====================================================

        cleanup_paths = (
            generated_clip_paths
            + temp_audio_files
        )

        if output_spot_path:
            cleanup_paths.append(
                output_spot_path
            )

        for path in cleanup_paths:

            try:
                if (
                    path
                    and os.path.exists(path)
                ):
                    os.remove(path)

            except Exception:
                pass

        gc.collect()
        torch.cuda.empty_cache()

        log_gpu_memory(
            "Fine job / cleanup"
        )


# ============================================================
# 13. START RUNPOD SERVERLESS
# ============================================================

runpod.serverless.start(
    {
        "handler": handler
    }
)
