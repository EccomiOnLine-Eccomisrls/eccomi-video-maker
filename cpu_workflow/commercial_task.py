import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from render import Retry, TaskContext
from supabase import create_client

OUT_W, OUT_H, FPS = 1080, 1920, 30
COMMERCIAL_PROTOCOL = "EVS_COMMERCIAL_PRODUCT_SERVICE_V1"
FONT_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
FONT_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]


def _download(url: str, path: Path) -> None:
    if not url:
        raise ValueError("URL_REQUIRED")
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFMPEG_FAILED: {proc.stderr[-3000:]}")


def _font(size: int, bold: bool = False):
    for path in (FONT_BOLD if bold else FONT_REGULAR):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    out = img.copy().convert("RGBA")
    out.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return out


def _gradient() -> Image.Image:
    top, bottom = (4, 19, 65, 255), (9, 68, 171, 255)
    base = Image.new("RGBA", (OUT_W, OUT_H), top)
    draw = ImageDraw.Draw(base)
    for y in range(OUT_H):
        a = y / max(1, OUT_H - 1)
        color = tuple(int(top[i] * (1 - a) + bottom[i] * a) for i in range(4))
        draw.line((0, y, OUT_W, y), fill=color)
    return base


def _rounded_card(img: Image.Image, max_w: int, max_h: int, radius: int = 36, pad: int = 16) -> Image.Image:
    inner = _fit(img, max_w - 2 * pad, max_h - 2 * pad)
    w, h = inner.width + 2 * pad, inner.height + 2 * pad
    shadow = Image.new("RGBA", (w + 38, h + 38), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((18, 18, w + 18, h + 18), radius=radius, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(13))
    card = Image.new("RGBA", shadow.size, (0, 0, 0, 0))
    card.alpha_composite(shadow)
    ImageDraw.Draw(card).rounded_rectangle((8, 8, w + 8, h + 8), radius=radius, fill=(255, 255, 255, 255))
    mask = Image.new("L", inner.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, inner.width, inner.height), radius=max(10, radius - pad), fill=255)
    card.paste(inner, (8 + pad, 8 + pad), mask)
    return card


def _draw_centered(draw: ImageDraw.ImageDraw, value: str, y: int, size: int, bold: bool = False, fill=(255, 255, 255, 255), max_width: int = 930) -> None:
    text = str(value or "").strip()
    if not text:
        return
    f = _font(size, bold)
    while size > 22:
        box = draw.textbbox((0, 0), text, font=f)
        if box[2] - box[0] <= max_width:
            break
        size -= 2
        f = _font(size, bold)
    box = draw.textbbox((0, 0), text, font=f)
    draw.text(((OUT_W - (box[2] - box[0])) // 2, y), text, font=f, fill=fill)


def _draw_multiline_centered(draw: ImageDraw.ImageDraw, value: str, y: int, size: int, max_width: int = 880, fill=(255, 255, 255, 255), bold: bool = False, line_gap: int = 12) -> None:
    words = str(value or "").strip().split()
    if not words:
        return
    f = _font(size, bold)
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if draw.textbbox((0, 0), test, font=f)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_h = size + line_gap
    for idx, line in enumerate(lines[:4]):
        box = draw.textbbox((0, 0), line, font=f)
        draw.text(((OUT_W - (box[2] - box[0])) // 2, y + idx * line_h), line, font=f, fill=fill)


def _paste_center(base: Image.Image, element: Image.Image, y: int, x_offset: int = 0) -> None:
    base.alpha_composite(element, ((OUT_W - element.width) // 2 + x_offset, y))


def _frame_logo(base: Image.Image, logo: Image.Image) -> None:
    base.alpha_composite(_fit(logo, 310, 110), (58, 55))


def _frame_hook(asset: Image.Image, logo: Image.Image, headline: str) -> Image.Image:
    base = _gradient()
    _frame_logo(base, logo)
    draw = ImageDraw.Draw(base)
    _draw_multiline_centered(draw, headline, 235, 62, max_width=940, bold=True)
    card = _rounded_card(asset, 930, 930)
    _paste_center(base, card, 700)
    return base


def _frame_asset(asset: Image.Image, logo: Image.Image, title: str, note: str = "") -> Image.Image:
    base = _gradient()
    _frame_logo(base, logo)
    draw = ImageDraw.Draw(base)
    _draw_multiline_centered(draw, title, 225, 48, max_width=930, bold=True)
    card = _rounded_card(asset, 980, 1040)
    _paste_center(base, card, 520)
    if note:
        _draw_multiline_centered(draw, note, 1605, 31, max_width=890, fill=(220, 235, 255, 255))
    return base


def _frame_offer(asset1: Image.Image, asset2: Image.Image, logo: Image.Image, offer: str) -> Image.Image:
    base = _gradient()
    _frame_logo(base, logo)
    draw = ImageDraw.Draw(base)
    _draw_centered(draw, "Semplice. Online. ECCOMI.", 220, 48, True)
    a = _rounded_card(asset1, 760, 650)
    b = _rounded_card(asset2, 600, 820)
    base.alpha_composite(a, (35, 540))
    base.alpha_composite(b, (OUT_W - b.width - 28, 830))
    if offer:
        draw.rounded_rectangle((145, 1565, 935, 1740), radius=82, fill=(255, 255, 255, 255))
        _draw_centered(draw, offer, 1612, 48, True, fill=(5, 39, 120, 255), max_width=720)
    return base


def _frame_final(asset: Image.Image, logo: Image.Image, title: str, offer: str, cta: str) -> Image.Image:
    base = _gradient()
    _frame_logo(base, logo)
    draw = ImageDraw.Draw(base)
    _draw_multiline_centered(draw, title, 225, 54, max_width=940, bold=True)
    card = _rounded_card(asset, 760, 780)
    _paste_center(base, card, 585)
    if offer:
        _draw_centered(draw, offer, 1435, 42, True, fill=(220, 235, 255, 255))
    draw.rounded_rectangle((105, 1605, 975, 1782), radius=88, fill=(255, 255, 255, 255))
    _draw_centered(draw, cta, 1657, 39, True, fill=(5, 39, 120, 255), max_width=800)
    return base


def _motion_segment(ffmpeg: str, image_path: Path, duration: float, output: Path, zoom_in: bool = True) -> None:
    if zoom_in:
        zoom = "min(zoom+0.0007,1.055)"
    else:
        zoom = "if(eq(on,1),1.055,max(1.0,zoom-0.0007))"
    vf = (
        f"zoompan=z='{zoom}':"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={OUT_W}x{OUT_H}:fps={FPS},format=yuv420p"
    )
    _run([
        ffmpeg, "-y", "-loop", "1", "-i", str(image_path),
        "-t", f"{duration:.3f}", "-vf", vf,
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-movflags", "+faststart", str(output),
    ])


def _concat_video(ffmpeg: str, segments: list[Path], output: Path) -> None:
    manifest = output.with_suffix(".txt")
    manifest.write_text("".join(f"file '{p.as_posix()}'\n" for p in segments), encoding="utf-8")
    _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(output)])


def _mix_audio(ffmpeg: str, visual: Path, voice: Path, music: Path, target_duration: float, output: Path, voice_volume: float, music_volume: float) -> None:
    filters = (
        f"[1:a]volume={voice_volume:.3f},apad,alimiter=limit=0.95[voice];"
        f"[2:a]volume={music_volume:.3f},apad[music];"
        "[voice][music]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[aout]"
    )
    _run([
        ffmpeg, "-y", "-i", str(visual), "-i", str(voice), "-stream_loop", "-1", "-i", str(music),
        "-filter_complex", filters, "-map", "0:v:0", "-map", "[aout]",
        "-t", f"{target_duration:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output),
    ])


def _callback(payload: dict[str, Any], body: dict[str, Any]) -> None:
    callback_url = str(payload.get("callback_url") or "").strip()
    anon_key = str(payload.get("supabase_anon_key") or "").strip()
    if not callback_url or not anon_key:
        raise ValueError("CALLBACK_CONFIG_MISSING")
    response = requests.post(
        callback_url,
        json=body,
        headers={
            "Authorization": f"Bearer {anon_key}",
            "apikey": anon_key,
            "Content-Type": "application/json",
            "User-Agent": "EVS-CPU-Commercial/1.0",
        },
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"CALLBACK_FAILED HTTP {response.status_code}: {response.text[:1200]}")


def register_commercial(app) -> None:
    @app.task(name="create_commercial", plan="flex", timeout_seconds=900, retry=Retry(max_retries=1, wait_duration_ms=1500, backoff_scaling=2.0))
    def create_commercial(ctx: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        evs_code = str(payload.get("evs_code") or "").strip()
        job_id = str(payload.get("job_id") or "").strip()
        asset1_url = str(payload.get("primary_asset_url") or "").strip()
        asset2_url = str(payload.get("secondary_asset_url") or "").strip()
        logo_url = str(payload.get("logo_url") or "").strip()
        voice_url = str(payload.get("voice_audio_url") or "").strip()
        music_url = str(payload.get("music_audio_url") or "").strip()
        output_path = str(payload.get("storage_path") or "").strip()
        upload_token = str(payload.get("storage_upload_token") or "").strip()
        output_public_url = str(payload.get("output_public_url") or "").strip()
        supabase_url = str(payload.get("supabase_url") or "").strip()
        anon_key = str(payload.get("supabase_anon_key") or "").strip()
        target_duration = max(8.0, min(30.0, float(payload.get("target_duration_seconds") or 15.0)))
        voice_volume = max(0.1, min(2.0, float(payload.get("voice_volume") or 1.05)))
        music_volume = max(0.0, min(1.0, float(payload.get("music_volume") or 0.20)))

        required = {
            "evs_code": evs_code,
            "job_id": job_id,
            "primary_asset_url": asset1_url,
            "secondary_asset_url": asset2_url,
            "logo_url": logo_url,
            "voice_audio_url": voice_url,
            "music_audio_url": music_url,
            "storage_path": output_path,
            "storage_upload_token": upload_token,
            "output_public_url": output_public_url,
            "supabase_url": supabase_url,
            "supabase_anon_key": anon_key,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError("MISSING_REQUIRED: " + ", ".join(missing))

        headline = str(payload.get("headline") or "Scoprilo con ECCOMI.").strip()
        scene2 = str(payload.get("scene_2_text") or "Tutto online, in pochi passaggi.").strip()
        scene3 = str(payload.get("scene_3_text") or "Completa la richiesta comodamente online.").strip()
        final_title = str(payload.get("final_title") or "ECCOMI ONLINE").strip()
        offer = str(payload.get("offer_text") or "").strip()
        cta = str(payload.get("cta_text") or "Scoprilo su Eccomi Online").strip()

        with tempfile.TemporaryDirectory(prefix="evs_commercial_") as tmpdir:
            root = Path(tmpdir)
            a1p, a2p, lp = root / "asset1", root / "asset2", root / "logo"
            vp, mp = root / "voice.wav", root / "music.wav"
            _download(asset1_url, a1p)
            _download(asset2_url, a2p)
            _download(logo_url, lp)
            _download(voice_url, vp)
            _download(music_url, mp)
            asset1 = Image.open(a1p).convert("RGBA")
            asset2 = Image.open(a2p).convert("RGBA")
            logo = Image.open(lp).convert("RGBA")

            frames = [
                _frame_hook(asset1, logo, headline),
                _frame_asset(asset1, logo, scene2),
                _frame_asset(asset2, logo, scene3),
                _frame_offer(asset1, asset2, logo, offer),
                _frame_final(asset2, logo, final_title, offer, cta),
            ]
            frame_paths: list[Path] = []
            for idx, frame in enumerate(frames):
                p = root / f"frame_{idx}.png"
                frame.save(p)
                frame_paths.append(p)

            durations = [2.5, 3.0, 3.0, 3.0, 3.5]
            factor = target_duration / sum(durations)
            durations = [round(x * factor, 3) for x in durations]
            durations[-1] += target_duration - sum(durations)

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            segments: list[Path] = []
            for idx, (frame_path, duration) in enumerate(zip(frame_paths, durations)):
                seg = root / f"seg_{idx}.mp4"
                _motion_segment(ffmpeg, frame_path, duration, seg, zoom_in=(idx % 2 == 0))
                segments.append(seg)

            visual = root / "visual.mp4"
            output = root / "commercial.mp4"
            _concat_video(ffmpeg, segments, visual)
            _mix_audio(ffmpeg, visual, vp, mp, target_duration, output, voice_volume, music_volume)

            supabase = create_client(supabase_url, anon_key)
            with output.open("rb") as fh:
                supabase.storage.from_("videos").upload_to_signed_url(path=output_path, token=upload_token, file=fh)

        elapsed = round(time.perf_counter() - started, 3)
        generation = {
            "mode": "cpu_commercial_product_service_non_gpu",
            "engine": "render_workflows_flex",
            "correction_protocol": COMMERCIAL_PROTOCOL,
            "timeline_seconds": durations,
            "scene_count": 5,
            "commercial_product_service": True,
            "mascot_used": False,
            "primary_asset_url": asset1_url,
            "secondary_asset_url": asset2_url,
            "voice_volume": voice_volume,
            "music_volume": music_volume,
            "total_seconds": elapsed,
        }
        qa = {
            "technical_pass": True,
            "output_width": OUT_W,
            "output_height": OUT_H,
            "target_duration_seconds": target_duration,
            "desktop_asset_composited": True,
            "mobile_asset_composited": True,
            "logo_composited_deterministically": True,
            "cta_composited_deterministically": True,
            "ai_brand_redraw": False,
            "cuts_only": False,
            "voice_present": True,
            "music_present": True,
            "release_gate_required": True,
            "cpu_route": True,
            "commercial_product_service": True,
            "mascot_used": False,
        }
        _callback(payload, {
            "event": "evs.video.completed",
            "status": "COMPLETED",
            "job_id": job_id,
            "spot_url": output_public_url,
            "customer_reference": evs_code,
            "generation": generation,
            "qa": qa,
        })
        return {
            "ok": True,
            "evs_code": evs_code,
            "job_id": job_id,
            "video_url": output_public_url,
            "processing_seconds": elapsed,
            "gpu_started": False,
            "route": "PRODUCT_SERVICE_CPU",
            "correction_protocol": COMMERCIAL_PROTOCOL,
        }
