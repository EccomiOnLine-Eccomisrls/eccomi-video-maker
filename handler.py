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
    "(Wan 2.1 14B - 80GB FULL GPU MODE)...",
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
# 2. CONFIG WAN
# ============================================================

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# ------------------------------------------------------------
# BENCHMARK:
# manteniamo gli stessi parametri usati sulla A40
# per confrontare correttamente velocità e costo.
# ------------------------------------------------------------

WAN_NUM_FRAMES = 33
WAN_NUM_STEPS = 30
WAN_GUIDANCE_SCALE = 5.0
EXPORT_FPS = 24


DEFAULT_SCENES = [
    (
        "ECCOMI MAN, the exact official illustrated superhero mascot "
        "from the reference image, confidently presenting the new "
        "ECCOMI ONLINE website. "
        "He performs one clear and controlled presenting gesture: "
        "his open presenting arm moves smoothly outward and slightly upward, "
        "his hand turns naturally toward the viewer, "
        "his head makes a small natural turn and returns toward the viewer, "
        "and his shoulders and upper torso shift subtly with the gesture. "
        "His red cape moves gently behind him. "
        "The movement must be clearly visible, smooth and professional. "
        "Preserve his face, hairstyle, body proportions, costume, "
        "red cape, red belt, gloves, boots and chest emblem. "
        "Keep both feet fixed. "
        "Fixed camera. No zoom. No scene change. "
        "No morphing. No extra fingers. No extra limbs."
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
    "--> [MODEL] Modalità FULL GPU 80GB.",
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

    # --------------------------------------------------------
    # FULL GPU LOAD
    #
    # device_map='cuda':
    # i componenti vengono caricati direttamente sulla GPU.
    #
    # low_cpu_mem_usage=True:
    # riduce il picco RAM durante il caricamento.
    # --------------------------------------------------------

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )

    # --------------------------------------------------------
    # VAE TILING / SLICING
    #
    # Manteniamo queste protezioni perché possono ridurre
    # i picchi durante encoding/decoding senza ricorrere
    # al disk offload.
    # --------------------------------------------------------

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

    # Il model card ufficiale mostra anche pipe.to("cuda").
    # Con device_map="cuda" dovrebbe essere già su GPU,
    # ma lo lasciamo come verifica esplicita.
    pipe.to("cuda")

    torch.cuda.empty_cache()
    gc.collect()

    model_load_seconds = (
        time.perf_counter() - model_load_start
    )

    print(
        f"✅ [MODEL] Wan 2.1 14B FULL GPU caricato "
        f"in {model_load_seconds:.1f} secondi.",
        flush=True,
    )

    log_gpu_memory("Dopo caricamento modello")

except Exception as e:
    print(
        "❌ [MODEL] ERRORE caricamento Wan FULL GPU:",
        flush=True,
    )

    print(
        str(e),
        flush=True,
    )

    # IMPORTANTE:
    # niente fallback automatico a CogVideoX o disk offload.
    #
    # Per questo benchmark vogliamo sapere chiaramente
    # se Wan 14B entra oppure no nella GPU 80GB.
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


# ============================================================
# 6. DOWNLOAD IMMAGINE
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
# 7. DOWNLOAD FILE
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
# 8. AUDIO
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
# 9. COSTRUZIONE PROMPT ECCOMI
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
        "one clear controlled gesture, "
        "natural arm motion, "
        "small coordinated shoulder movement, "
        "subtle upper torso movement, "
        "small natural head motion, "
        "gentle cape movement, "
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
# 10. GENERAZIONE SINGOLA CLIP WAN
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
        f"✅ [WAN] Inferenza completata in "
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

    # Liberiamo esplicitamente i frame PIL
    del frames
    del result

    gc.collect()
    torch.cuda.empty_cache()

    return temp_video_path


# ============================================================
# 11. HANDLER RUNPOD
# ============================================================

def handler(event):

    job_start = time.perf_counter()

    print(
        "--- 🚀 ECCOMI VIDEO MAKER "
        "| WAN 2.1 14B "
        "| FULL GPU 80GB ---",
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

    if not image_url:
        return {
            "error": "Nessuna image_url fornita."
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

        # NOTA:
        # NON facciamo ImageOps.fit/crop.
        # Manteniamo l'immagine originale e lasciamo
        # il preprocessing alla pipeline Wan.
        # Questo evita di tagliare ECCOMI MAN.

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
            f"✅ [RENDER] Completato in "
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

        safe_job_id = (
            str(job_id)
            .replace("/", "_")
            .replace("\\", "_")
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

        print(
            "============================================",
            flush=True,
        )

        print(
            "✅ ECCOMI VIDEO COMPLETATO",
            flush=True,
        )

        print(
            f"✅ TEMPO TOTALE JOB: "
            f"{total_job_seconds:.1f} sec "
            f"({total_job_seconds / 60:.2f} min)",
            flush=True,
        )

        print(
            f"✅ URL VIDEO: {public_video_url}",
            flush=True,
        )

        print(
            "============================================",
            flush=True,
        )

        return {
            "spot_url": public_video_url,
            "generation": {
                "model": MODEL_ID,
                "mode": "full_gpu_80gb",
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
                "frames": WAN_NUM_FRAMES,
                "steps": WAN_NUM_STEPS,
                "guidance_scale": WAN_GUIDANCE_SCALE,
                "fps": EXPORT_FPS,
                "total_seconds": round(
                    total_job_seconds,
                    2,
                ),
            },
        }

    except torch.cuda.OutOfMemoryError as e:

        print(
            "❌ [CUDA OOM] Memoria GPU insufficiente.",
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

        return {
            "error": "CUDA_OUT_OF_MEMORY",
            "details": str(e),
        }

    except Exception as e:

        print(
            f"❌ [ERROR] {type(e).__name__}: {e}",
            flush=True,
        )

        log_gpu_memory(
            "Errore job"
        )

        gc.collect()
        torch.cuda.empty_cache()

        return {
            "error": str(e)
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
# 12. START RUNPOD SERVERLESS
# ============================================================

runpod.serverless.start(
    {
        "handler": handler
    }
)
