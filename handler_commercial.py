import gc
import os
import tempfile
import time
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import requests
import runpod
from supabase import Client, create_client
import torch
import ftfy

from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
import moviepy.audio.fx.all as afx

print("--> [INIT] EVS COMMERCIAL WORKER | Wan 2.1 14B + deterministic compositor", flush=True)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
BASE_SUPABASE_URL = SUPABASE_URL.rstrip("/")
EVS_CALLBACK_URL = os.environ.get(
    "EVS_CALLBACK_URL",
    f"{BASE_SUPABASE_URL}/functions/v1/evs-video-callback" if BASE_SUPABASE_URL else "",
).strip()
EVS_CALLBACK_TOKEN = os.environ.get("EVS_CALLBACK_TOKEN", "").strip()
CALLBACK_TIMEOUT_SECONDS = 20
CALLBACK_MAX_ATTEMPTS = 3

if not BASE_SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_KEY non configurati.")

supabase: Client = create_client(BASE_SUPABASE_URL, SUPABASE_KEY)

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
WAN_WIDTH = 1280
WAN_HEIGHT = 720
WAN_FRAMES = 49
WAN_STEPS = 30
WAN_GUIDANCE = 5.0
FPS = 24
OUT_W = 1080
OUT_H = 1920

DEFAULT_SCENES = [
    (
        "Professional illustrated brand mascot, same exact identity as reference, "
        "confident presenter gesture toward the product area, controlled upper-body motion, "
        "fixed camera, stable face, stable costume, stable proportions, no morphing, "
        "no extra fingers, no extra hands, no extra limbs."
    )
]

FONT_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
FONT_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]


def font(size: int, bold: bool = False):
    for path in (FONT_BOLD if bold else FONT_REGULAR):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def clean(value) -> str:
    return ftfy.fix_text(str(value or "")).strip()


def tmp(suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = f.name
    f.close()
    return path


def safe_job_id(value) -> str:
    return str(value).replace("/", "_").replace("\\", "_")


def log_gpu(stage: str):
    if not torch.cuda.is_available():
        return
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    print(
        f"--> [GPU] {stage} | {p.name} | "
        f"alloc={torch.cuda.memory_allocated(i)/(1024**3):.2f}GiB | "
        f"reserved={torch.cuda.memory_reserved(i)/(1024**3):.2f}GiB | "
        f"peak={torch.cuda.max_memory_allocated(i)/(1024**3):.2f}GiB",
        flush=True,
    )


def send_callback(callback_url: str, payload: dict) -> dict:
    callback_url = clean(callback_url)
    if not callback_url:
        return {"enabled": False, "delivered": False, "status": "SKIPPED"}
    headers = {"Content-Type": "application/json", "User-Agent": "EVS-PRO-Commercial/1.0"}
    token = EVS_CALLBACK_TOKEN or SUPABASE_KEY
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if SUPABASE_KEY:
        headers["apikey"] = SUPABASE_KEY
    last = None
    for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
        try:
            r = requests.post(callback_url, json=payload, headers=headers, timeout=CALLBACK_TIMEOUT_SECONDS)
            if 200 <= r.status_code < 300:
                return {
                    "enabled": True,
                    "delivered": True,
                    "status": "DELIVERED",
                    "http_status": r.status_code,
                    "attempts": attempt,
                }
            last = f"HTTP {r.status_code}: {r.text[:500]}"
        except Exception as e:
            last = str(e)
        if attempt < CALLBACK_MAX_ATTEMPTS:
            time.sleep(min(2 ** (attempt - 1), 4))
    return {
        "enabled": True,
        "delivered": False,
        "status": "FAILED",
        "attempts": CALLBACK_MAX_ATTEMPTS,
        "error": last,
    }


def fetch_image(url: str, rgba: bool = True) -> Image.Image:
    r = requests.get(clean(url), timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGBA" if rgba else "RGB")


def fetch_file(url: str, path: str):
    r = requests.get(clean(url), timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def fetch_audio(url: str, path: str):
    raw = tmp("")
    fetch_file(url, raw)
    u = clean(url).lower().split("?")[0].split("#")[0]
    is_video = any(u.endswith(x) for x in [".mov", ".mp4", ".avi", ".mkv", ".webm", ".quicktime"])
    if not is_video:
        if os.path.exists(path):
            os.remove(path)
        os.rename(raw, path)
        return
    v = None
    try:
        v = VideoFileClip(raw)
        if v.audio is None:
            raise RuntimeError("Il file video non contiene audio.")
        v.audio.write_audiofile(path, logger=None, fps=44100)
    finally:
        if v is not None:
            try:
                v.close()
            except Exception:
                pass
        if os.path.exists(raw):
            os.remove(raw)


def validate_image(img: Image.Image, name: str):
    if img.width < 80 or img.height < 80:
        raise ValueError(f"{name}_TOO_SMALL:{img.width}x{img.height}")


def fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    out = img.copy()
    out.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return out


def gradient_background(w=OUT_W, h=OUT_H):
    top = (5, 25, 86, 255)
    bottom = (7, 66, 180, 255)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        a = y / max(1, h - 1)
        arr[y, :, 0] = int(top[0] * (1-a) + bottom[0] * a)
        arr[y, :, 1] = int(top[1] * (1-a) + bottom[1] * a)
        arr[y, :, 2] = int(top[2] * (1-a) + bottom[2] * a)
        arr[y, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


def rounded_card(img: Image.Image, max_w: int, max_h: int, radius=34, pad=16):
    inner = fit(img.convert("RGBA"), max_w - 2*pad, max_h - 2*pad)
    w = inner.width + 2*pad
    h = inner.height + 2*pad
    shadow = Image.new("RGBA", (w + 34, h + 34), (0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((18, 18, w+18, h+18), radius=radius, fill=(0,0,0,105))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    card = Image.new("RGBA", shadow.size, (0,0,0,0))
    card.alpha_composite(shadow)
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((8, 8, w+8, h+8), radius=radius, fill=(255,255,255,255))
    mask = Image.new("L", inner.size, 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0,0,inner.width,inner.height), radius=max(8, radius-pad), fill=255)
    card.paste(inner, (8+pad, 8+pad), mask)
    return card


def draw_centered(draw, text, y, size, bold=False, fill=(255,255,255,255), max_width=940):
    text = clean(text)
    if not text:
        return y
    f = font(size, bold)
    while size > 22:
        box = draw.textbbox((0,0), text, font=f)
        if box[2]-box[0] <= max_width:
            break
        size -= 2
        f = font(size, bold)
    box = draw.textbbox((0,0), text, font=f)
    x = (OUT_W - (box[2]-box[0])) // 2
    draw.text((x,y), text, font=f, fill=fill)
    return y + (box[3]-box[1])


def paste_center(base: Image.Image, element: Image.Image, y: int, x_offset=0):
    x = (base.width - element.width)//2 + x_offset
    base.alpha_composite(element, (x, y))


def make_frame(kind: str, desktop: Image.Image, mobile: Image.Image, logo: Image.Image, brand: str, headline: str, cta: str):
    base = gradient_background()
    d = ImageDraw.Draw(base)
    lg = fit(logo, 310, 110)
    base.alpha_composite(lg, (60, 60))
    if kind == "opener":
        draw_centered(d, headline or f"{brand} cambia look!", 190, 62, True, max_width=950)
        card = rounded_card(desktop, 930, 590)
        paste_center(base, card, 1120)
        draw_centered(d, "WEB + MOBILE", 1725, 34, True, fill=(210,230,255,255))
    elif kind == "desktop":
        draw_centered(d, "Un nuovo modo di scoprire tutto.", 220, 48, True, max_width=950)
        card = rounded_card(desktop, 980, 760)
        paste_center(base, card, 500)
        draw_centered(d, "Desktop reale", 1360, 34, True, fill=(210,230,255,255))
    elif kind == "mobile":
        draw_centered(d, "Tutto il mondo ECCOMI. Anche mobile.", 205, 48, True, max_width=960)
        card = rounded_card(mobile, 520, 1040)
        paste_center(base, card, 500)
        draw_centered(d, "Mobile reale", 1580, 34, True, fill=(210,230,255,255))
    elif kind == "both":
        draw_centered(d, "Web e mobile. In un solo clic.", 205, 50, True, max_width=960)
        desk = rounded_card(desktop, 820, 540)
        mob = rounded_card(mobile, 360, 760)
        base.alpha_composite(desk, (60, 520))
        base.alpha_composite(mob, (665, 880))
    else:
        draw_centered(d, brand or "ECCOMI ONLINE", 220, 58, True, max_width=960)
        desk = rounded_card(desktop, 720, 470)
        mob = rounded_card(mobile, 310, 650)
        base.alpha_composite(desk, (55, 520))
        base.alpha_composite(mob, (710, 760))
        pill = Image.new("RGBA", (920, 180), (0,0,0,0))
        pd = ImageDraw.Draw(pill)
        pd.rounded_rectangle((0,0,920,180), radius=90, fill=(255,255,255,255))
        txt = clean(cta) or "Scoprilo ora"
        fs = 46
        pf = font(fs, True)
        while fs > 28:
            bb = pd.textbbox((0,0), txt, font=pf)
            if bb[2]-bb[0] < 820:
                break
            fs -= 2
            pf = font(fs, True)
        bb = pd.textbbox((0,0), txt, font=pf)
        pd.text(((920-(bb[2]-bb[0]))//2, (180-(bb[3]-bb[1]))//2-4), txt, font=pf, fill=(5,39,120,255))
        paste_center(base, pill, 1600)
    return base


def enhance_prompt(prompt: str) -> str:
    return (
        f"{clean(prompt)}, premium comic illustration, professional branded mascot animation, "
        "same character identity as input, stable recognizable face, stable hairstyle, "
        "stable body proportions, stable costume, smooth temporal consistency, fixed camera, "
        "no character redesign, no face morphing, no costume morphing, no extra arms, "
        "no extra hands, no extra fingers, no extra legs, no body distortion"
    )


if not torch.cuda.is_available():
    raise RuntimeError("CUDA non disponibile: EVS commercial worker richiede GPU NVIDIA.")

from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

print(f"--> [MODEL] Caricamento {MODEL_ID} su CUDA...", flush=True)
model_started = time.perf_counter()
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True,
)
try:
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
except Exception:
    pass
pipe.to("cuda")
gc.collect()
torch.cuda.empty_cache()
print(f"--> [MODEL] Caricato in {time.perf_counter()-model_started:.1f}s", flush=True)


def generate_clip(init_image: Image.Image, prompt: str) -> str:
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    log_gpu("prima inferenza")
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            image=init_image.convert("RGB"),
            prompt=enhance_prompt(prompt),
            height=WAN_HEIGHT,
            width=WAN_WIDTH,
            num_frames=WAN_FRAMES,
            num_inference_steps=WAN_STEPS,
            guidance_scale=WAN_GUIDANCE,
        )
    frames = result.frames[0]
    path = tmp(".mp4")
    export_to_video(frames, path, fps=FPS)
    del frames, result
    gc.collect()
    torch.cuda.empty_cache()
    print(f"--> [WAN] scena completata in {time.perf_counter()-started:.1f}s", flush=True)
    return path


def mascot_band(path: str, duration: float, y=400, height=520):
    clip = VideoFileClip(path).without_audio()
    usable = min(duration, max(0.1, clip.duration - 0.02))
    clip = clip.subclip(0, usable).resize(width=OUT_W)
    h = min(height, int(clip.h))
    clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=OUT_W, height=h)
    return clip.set_position((0, y)), usable


def commercial_timeline(desktop, mobile, logo, brand, headline, cta, generated_paths, target_duration):
    base_durations = [2.0, 3.4, 3.2, 3.2, 3.2]
    factor = target_duration / sum(base_durations)
    durations = [round(x*factor, 3) for x in base_durations]
    durations[-1] += target_duration - sum(durations)
    kinds = ["opener", "desktop", "mobile", "both", "cta"]
    segments = []
    temp_clips = []
    for kind, dur in zip(kinds, durations):
        frame = make_frame(kind, desktop, mobile, logo, brand, headline, cta)
        bg = ImageClip(np.array(frame)).set_duration(dur)
        if kind == "opener" and generated_paths:
            band, _ = mascot_band(generated_paths[0], dur, y=430, height=520)
            temp_clips.append(band)
            seg = CompositeVideoClip([bg, band], size=(OUT_W, OUT_H)).set_duration(dur)
        elif kind == "cta" and len(generated_paths) > 1:
            band, _ = mascot_band(generated_paths[-1], min(dur, 2.0), y=1040, height=420)
            temp_clips.append(band)
            seg = CompositeVideoClip([bg, band], size=(OUT_W, OUT_H)).set_duration(dur)
        else:
            seg = bg
        segments.append(seg)
    final = concatenate_videoclips(segments, method="compose").set_duration(target_duration)
    return final, segments, temp_clips, durations


def legacy_timeline(paths):
    clips = [VideoFileClip(p) for p in paths]
    if len(clips) == 1:
        return clips[0], clips
    return concatenate_videoclips(clips, method="compose"), clips


def handler(event):
    started = time.perf_counter()
    inp = event.get("input", {}) or {}
    job_id = str(event.get("id", "test_job"))
    callback_url = clean(inp.get("callback_url") or EVS_CALLBACK_URL)
    reference = clean(inp.get("customer_reference"))
    mascot_url = clean(inp.get("image_url"))
    voice_url = clean(inp.get("voice_audio_url"))
    music_url = clean(inp.get("music_audio_url"))
    prompts = inp.get("scenes_prompts") or DEFAULT_SCENES
    commercial_mode = bool(inp.get("commercial_mode", True))
    target_duration = float(inp.get("target_duration_seconds") or 15.0)
    target_duration = max(8.0, min(target_duration, 30.0))

    desktop_url = clean(inp.get("desktop_asset_url"))
    mobile_url = clean(inp.get("mobile_asset_url"))
    logo_url = clean(inp.get("logo_url"))
    cta = clean(inp.get("cta_text"))
    brand = clean(inp.get("brand_name"))
    headline = clean(inp.get("headline")) or (f"{brand} cambia look!" if brand else "Un nuovo modo di scoprirti.")
    music_volume = float(inp.get("music_volume") or 0.20)
    voice_volume = float(inp.get("voice_volume") or 1.0)

    generated = []
    temp_audio = []
    opened_video = []
    opened_audio = []
    output = None
    final = None
    final_audio = None
    extra_segments = []
    extra_clips = []

    def fail(error: str, details: str = ""):
        payload = {
            "event": "evs.video.failed",
            "status": "FAILED",
            "job_id": job_id,
            "customer_reference": reference or None,
            "error": error,
            "details": details or None,
            "commercial_mode": commercial_mode,
        }
        payload["delivery"] = send_callback(callback_url, payload)
        return payload

    try:
        if not mascot_url:
            return fail("MISSING_MASCOT_IMAGE")

        desktop = mobile = logo = None
        if commercial_mode:
            missing = [
                name for name, value in [
                    ("desktop_asset_url", desktop_url),
                    ("mobile_asset_url", mobile_url),
                    ("logo_url", logo_url),
                    ("cta_text", cta),
                ] if not value
            ]
            if missing:
                return fail("COMMERCIAL_PREFLIGHT_FAILED", "Missing: " + ", ".join(missing))
            desktop = fetch_image(desktop_url)
            mobile = fetch_image(mobile_url)
            logo = fetch_image(logo_url)
            validate_image(desktop, "DESKTOP_ASSET")
            validate_image(mobile, "MOBILE_ASSET")
            validate_image(logo, "LOGO")
            print(f"--> [PREFLIGHT] desktop={desktop.size} mobile={mobile.size} logo={logo.size} CTA=OK", flush=True)

        init_image = fetch_image(mascot_url, rgba=False)
        validate_image(init_image, "MASCOT")

        if not isinstance(prompts, list) or not prompts:
            prompts = DEFAULT_SCENES
        prompts = [clean(x) for x in prompts if clean(x)] or DEFAULT_SCENES

        for i, prompt in enumerate(prompts):
            print(f"--> [SCENA] {i+1}/{len(prompts)}", flush=True)
            generated.append(generate_clip(init_image, prompt))

        if commercial_mode:
            final_base, extra_segments, extra_clips, timeline_durations = commercial_timeline(
                desktop, mobile, logo, brand, headline, cta, generated, target_duration
            )
        else:
            final_base, opened_video = legacy_timeline(generated)
            timeline_durations = [round(final_base.duration, 3)]

        audio_layers = []
        if voice_url:
            vp = tmp(".wav")
            fetch_audio(voice_url, vp)
            temp_audio.append(vp)
            vc = AudioFileClip(vp).volumex(voice_volume)
            vc = vc.subclip(0, min(vc.duration, target_duration))
            opened_audio.append(vc)
            audio_layers.append(vc)

        if music_url:
            mp = tmp(".wav")
            fetch_audio(music_url, mp)
            temp_audio.append(mp)
            mc0 = AudioFileClip(mp).volumex(music_volume)
            opened_audio.append(mc0)
            mc = afx.audio_loop(mc0, duration=target_duration)
            audio_layers.append(mc)

        if audio_layers:
            final_audio = CompositeAudioClip(audio_layers).set_duration(target_duration)
            final = final_base.set_audio(final_audio)
        else:
            final = final_base

        output = tmp(".mp4")
        print("--> [RENDER] Master commerciale...", flush=True)
        final.write_videofile(
            output,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            bitrate="9000k",
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            logger=None,
        )

        object_path = f"{safe_job_id(job_id)}_commercial_master.mp4" if commercial_mode else f"{safe_job_id(job_id)}_spot_wan21.mp4"
        with open(output, "rb") as fh:
            supabase.storage.from_("videos").upload(
                path=object_path,
                file=fh,
                file_options={"content-type": "video/mp4", "upsert": "true"},
            )
        public_url = f"{BASE_SUPABASE_URL}/storage/v1/object/public/videos/{object_path}"
        elapsed = time.perf_counter() - started

        technical_qa = {
            "technical_pass": True,
            "commercial_mode": commercial_mode,
            "output_width": OUT_W if commercial_mode else WAN_WIDTH,
            "output_height": OUT_H if commercial_mode else WAN_HEIGHT,
            "target_duration_seconds": round(target_duration, 2),
            "desktop_asset_composited": bool(commercial_mode and desktop_url),
            "mobile_asset_composited": bool(commercial_mode and mobile_url),
            "logo_composited_deterministically": bool(commercial_mode and logo_url),
            "cta_composited_deterministically": bool(commercial_mode and cta),
            "ai_brand_redraw": False,
            "cuts_only": True,
            "voice_present": bool(voice_url),
            "music_present": bool(music_url),
            "release_gate_required": True,
        }
        generation = {
            "model": MODEL_ID,
            "mode": "full_gpu_80gb_commercial" if commercial_mode else "full_gpu_80gb",
            "wan_width": WAN_WIDTH,
            "wan_height": WAN_HEIGHT,
            "frames": WAN_FRAMES,
            "steps": WAN_STEPS,
            "guidance_scale": WAN_GUIDANCE,
            "fps": FPS,
            "scene_count": len(generated),
            "timeline_seconds": timeline_durations,
            "total_seconds": round(elapsed, 2),
        }
        payload = {
            "event": "evs.video.completed",
            "status": "COMPLETED",
            "job_id": job_id,
            "customer_reference": reference or None,
            "spot_url": public_url,
            "generation": generation,
            "qa": technical_qa,
            "commercial_mode": commercial_mode,
        }
        payload["delivery"] = send_callback(callback_url, payload)
        return payload

    except torch.cuda.OutOfMemoryError as e:
        gc.collect()
        torch.cuda.empty_cache()
        return fail("CUDA_OUT_OF_MEMORY", str(e))
    except Exception as e:
        return fail(type(e).__name__, str(e))
    finally:
        try:
            if final_audio is not None:
                final_audio.close()
        except Exception:
            pass
        try:
            if final is not None:
                final.close()
        except Exception:
            pass
        for clip in opened_audio:
            try:
                clip.close()
            except Exception:
                pass
        for clip in opened_video + extra_segments + extra_clips:
            try:
                clip.close()
            except Exception:
                pass
        for p in generated + temp_audio + ([output] if output else []):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu("fine job")


runpod.serverless.start({"handler": handler})
