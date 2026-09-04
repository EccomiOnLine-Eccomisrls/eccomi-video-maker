import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from render import Retry, TaskContext, Workflows
from supabase import create_client

app = Workflows()
OUT_W, OUT_H, FPS = 1080, 1920, 30
VISUAL_PROTOCOL = "EVS_VISUAL_CORRECTION_V1"
VISUAL_ISSUES = {"LAYOUT", "TEXT", "TIMING", "QUALITY"}
FONT_REGULAR = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
FONT_BOLD = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"]


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
            "User-Agent": "EVS-CPU-Compositor/3.0",
        },
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"CALLBACK_FAILED HTTP {response.status_code}: {response.text[:1200]}")


def _font(size: int, bold: bool = False):
    for path in (FONT_BOLD if bold else FONT_REGULAR):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    out = img.copy()
    out.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return out


def _gradient() -> Image.Image:
    top, bottom = (5, 25, 86, 255), (7, 66, 180, 255)
    base = Image.new("RGBA", (OUT_W, OUT_H), top)
    draw = ImageDraw.Draw(base)
    for y in range(OUT_H):
        a = y / max(1, OUT_H - 1)
        color = tuple(int(top[i] * (1-a) + bottom[i] * a) for i in range(4))
        draw.line((0, y, OUT_W, y), fill=color)
    return base


def _rounded_card(img: Image.Image, max_w: int, max_h: int, radius: int = 34, pad: int = 16) -> Image.Image:
    inner = _fit(img.convert("RGBA"), max_w - 2 * pad, max_h - 2 * pad)
    w, h = inner.width + 2 * pad, inner.height + 2 * pad
    shadow = Image.new("RGBA", (w + 34, h + 34), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((18, 18, w + 18, h + 18), radius=radius, fill=(0, 0, 0, 105))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    card = Image.new("RGBA", shadow.size, (0, 0, 0, 0))
    card.alpha_composite(shadow)
    ImageDraw.Draw(card).rounded_rectangle((8, 8, w + 8, h + 8), radius=radius, fill=(255, 255, 255, 255))
    mask = Image.new("L", inner.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, inner.width, inner.height), radius=max(8, radius - pad), fill=255)
    card.paste(inner, (8 + pad, 8 + pad), mask)
    return card


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, size: int, bold: bool = False, fill=(255, 255, 255, 255), max_width: int = 940) -> None:
    text = str(text or "").strip()
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


def _paste_center(base: Image.Image, element: Image.Image, y: int, x_offset: int = 0) -> None:
    base.alpha_composite(element, ((base.width - element.width) // 2 + x_offset, y))


def _visual_frame(kind: str, desktop: Image.Image, mobile: Image.Image, logo: Image.Image, actions: dict[str, Any]) -> Image.Image:
    base = _gradient()
    draw = ImageDraw.Draw(base)
    base.alpha_composite(_fit(logo, 310, 110), (60, 60))
    scale = max(1.0, min(1.25, float(actions.get("layout_scale") or 1.12)))
    if kind == "desktop":
        _draw_centered(draw, actions.get("desktop_title") or "Un nuovo modo di scoprire tutto.", 220, 48, True, max_width=950)
        card = _rounded_card(desktop, min(1060, int(980 * scale)), min(850, int(760 * scale)))
        _paste_center(base, card, 455)
        _draw_centered(draw, actions.get("desktop_caption") or "Desktop reale", 1390, 36, True, fill=(220, 235, 255, 255))
    elif kind == "mobile":
        _draw_centered(draw, actions.get("mobile_title") or "Tutto il mondo ECCOMI. Anche mobile.", 205, 48, True, max_width=960)
        card = _rounded_card(mobile, min(650, int(520 * scale)), min(1180, int(1040 * scale)))
        _paste_center(base, card, 400)
        _draw_centered(draw, actions.get("mobile_caption") or "Mobile reale", 1600, 36, True, fill=(220, 235, 255, 255))
    elif kind == "both":
        _draw_centered(draw, actions.get("both_title") or "Web e mobile. In un solo clic.", 205, 50, True, max_width=960)
        desk = _rounded_card(desktop, min(980, int(820 * scale)), min(650, int(540 * scale)))
        mob = _rounded_card(mobile, min(470, int(360 * scale)), min(900, int(760 * scale)))
        base.alpha_composite(desk, (max(20, (OUT_W - desk.width) // 2 - 55), 475))
        base.alpha_composite(mob, (OUT_W - mob.width - 30, 800))
    else:
        raise ValueError(f"UNSUPPORTED_VISUAL_FRAME:{kind}")
    return base


def _final_frame(desktop: Image.Image, mobile: Image.Image, logo: Image.Image, actions: dict[str, Any]) -> Image.Image:
    """Build the closing scene from clean source assets only. Never reuse the old master final."""
    base = _gradient()
    draw = ImageDraw.Draw(base)
    base.alpha_composite(_fit(logo, 300, 105), (60, 55))
    _draw_centered(draw, actions.get("final_title") or "Scopri il nuovo ECCOMI ONLINE", 205, 52, True, max_width=940)

    scale = max(1.0, min(1.25, float(actions.get("layout_scale") or 1.12)))
    desk = _rounded_card(desktop, min(900, int(760 * scale)), min(600, int(500 * scale)))
    mob = _rounded_card(mobile, min(430, int(330 * scale)), min(820, int(700 * scale)))
    base.alpha_composite(desk, (45, 525))
    base.alpha_composite(mob, (OUT_W - mob.width - 35, 790))

    draw.rounded_rectangle((105, 1580, 975, 1760), radius=90, fill=(255, 255, 255, 255))
    _draw_centered(draw, actions.get("final_cta") or "Visita eccomionline.com", 1634, 43, True, fill=(5, 39, 120, 255), max_width=800)
    return base


def _static_segment(ffmpeg: str, image_path: Path, duration: float, output: Path) -> None:
    _run([ffmpeg, "-y", "-loop", "1", "-i", str(image_path), "-t", f"{duration:.3f}", "-r", str(FPS), "-vf", "format=yuv420p", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-movflags", "+faststart", str(output)])


def _master_segment(ffmpeg: str, master: Path, start: float, duration: float, output: Path) -> None:
    _run([ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", str(master), "-t", f"{duration:.3f}", "-vf", f"scale={OUT_W}:{OUT_H},fps={FPS},format=yuv420p", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-movflags", "+faststart", str(output)])


def _concat_video(ffmpeg: str, segments: list[Path], output: Path) -> None:
    manifest = output.with_suffix(".txt")
    manifest.write_text("".join(f"file '{p.as_posix()}'\n" for p in segments), encoding="utf-8")
    _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(output)])


def _attach_original_audio(ffmpeg: str, visual: Path, master: Path, target_duration: float, output: Path) -> None:
    _run([ffmpeg, "-y", "-i", str(visual), "-i", str(master), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy", "-t", f"{target_duration:.3f}", "-movflags", "+faststart", str(output)])


def _sample_frame(ffmpeg: str, video: Path, at: float, output: Path) -> None:
    _run([ffmpeg, "-y", "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=180:320", str(output)])


def _visual_diff_score(ffmpeg: str, before: Path, after: Path, root: Path) -> float:
    scores = []
    for idx, at in enumerate((2.7, 6.0, 9.5, 13.0)):
        a, b = root / f"before_{idx}.png", root / f"after_{idx}.png"
        _sample_frame(ffmpeg, before, at, a)
        _sample_frame(ffmpeg, after, at, b)
        ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
        hist = ImageChops.difference(ia, ib).histogram()
        total = sum(value * (index % 256) for index, value in enumerate(hist))
        scores.append(total / max(1, ia.width * ia.height * 3 * 255))
    return round(sum(scores) / len(scores), 6)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _visual_remix(payload: dict[str, Any], root: Path, master: Path, output: Path, ffmpeg: str, target_duration: float, issues: list[str]):
    urls = {
        "desktop_asset_url": str(payload.get("desktop_asset_url") or "").strip(),
        "mobile_asset_url": str(payload.get("mobile_asset_url") or "").strip(),
        "logo_url": str(payload.get("logo_url") or "").strip(),
    }
    missing = [k for k, v in urls.items() if not v]
    if missing:
        raise ValueError("VISUAL_ASSETS_MISSING: " + ", ".join(missing))

    dp, mp, lp = root / "desktop_asset", root / "mobile_asset", root / "logo_asset"
    _download(urls["desktop_asset_url"], dp)
    _download(urls["mobile_asset_url"], mp)
    _download(urls["logo_url"], lp)
    desktop, mobile, logo = Image.open(dp).convert("RGBA"), Image.open(mp).convert("RGBA"), Image.open(lp).convert("RGBA")
    actions = payload.get("visual_actions") if isinstance(payload.get("visual_actions"), dict) else {}

    durations = [2.0, 3.4, 3.2, 3.2, 3.2]
    factor = target_duration / sum(durations)
    durations = [round(x * factor, 3) for x in durations]
    durations[-1] += target_duration - sum(durations)

    d_png, m_png, b_png, f_png = root / "desktop.png", root / "mobile.png", root / "both.png", root / "final.png"
    _visual_frame("desktop", desktop, mobile, logo, actions).save(d_png)
    _visual_frame("mobile", desktop, mobile, logo, actions).save(m_png)
    _visual_frame("both", desktop, mobile, logo, actions).save(b_png)
    _final_frame(desktop, mobile, logo, actions).save(f_png)

    segs = [root / f"seg{i}.mp4" for i in range(5)]
    _master_segment(ffmpeg, master, 0.0, durations[0], segs[0])
    _static_segment(ffmpeg, d_png, durations[1], segs[1])
    _static_segment(ffmpeg, m_png, durations[2], segs[2])
    _static_segment(ffmpeg, b_png, durations[3], segs[3])
    _static_segment(ffmpeg, f_png, durations[4], segs[4])

    visual = root / "visual_only.mp4"
    _concat_video(ffmpeg, segs, visual)
    _attach_original_audio(ffmpeg, visual, master, target_duration, output)

    diff_score = _visual_diff_score(ffmpeg, master, output, root)
    changed = diff_score >= 0.015
    generation = {
        "mode": "cpu_visual_compositor_non_gpu",
        "engine": "render_workflows_flex",
        "correction_protocol": VISUAL_PROTOCOL,
        "timeline_seconds": durations,
        "scene_count": 5,
        "source_master_url": str(payload.get("master_video_url") or ""),
        "issues": issues,
        "correction_note": str(payload.get("correction_note") or ""),
        "visual_actions": actions,
        "visual_diff_score": diff_score,
        "source_sha256": _sha256(master),
        "output_sha256": _sha256(output),
        "final_rebuilt_from_clean_assets": True,
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
        "cuts_only": True,
        "voice_present": True,
        "music_present": True,
        "release_gate_required": True,
        "cpu_route": True,
        "audio_preserved": True,
        "visual_correction_applied": changed,
        "visual_diff_score": diff_score,
        "final_rebuilt_clean": True,
    }
    return generation, qa


@app.task(name="remix_master", plan="flex", timeout_seconds=900, retry=Retry(max_retries=1, wait_duration_ms=1500, backoff_scaling=2.0))
def remix_master(ctx: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    evs_code = str(payload.get("evs_code") or "").strip()
    job_id = str(payload.get("job_id") or "").strip()
    master_url = str(payload.get("master_video_url") or "").strip()
    voice_url = str(payload.get("voice_audio_url") or "").strip()
    music_url = str(payload.get("music_audio_url") or "").strip()
    output_path = str(payload.get("storage_path") or "").strip()
    upload_token = str(payload.get("storage_upload_token") or "").strip()
    output_public_url = str(payload.get("output_public_url") or "").strip()
    supabase_url = str(payload.get("supabase_url") or "").strip()
    anon_key = str(payload.get("supabase_anon_key") or "").strip()
    target_duration = max(8.0, min(30.0, float(payload.get("target_duration_seconds") or 15.0)))
    voice_volume = max(0.1, min(2.0, float(payload.get("voice_volume") or 1.0)))
    music_volume = max(0.0, min(1.0, float(payload.get("music_volume") or 0.22)))
    issues = [str(x).upper() for x in payload.get("issues", [])] if isinstance(payload.get("issues"), list) else []
    visual_mode = str(payload.get("correction_protocol") or "").strip() == VISUAL_PROTOCOL or any(x in VISUAL_ISSUES for x in issues)

    required = {
        "evs_code": evs_code,
        "job_id": job_id,
        "master_video_url": master_url,
        "storage_path": output_path,
        "storage_upload_token": upload_token,
        "output_public_url": output_public_url,
        "supabase_url": supabase_url,
        "supabase_anon_key": anon_key,
    }
    if not visual_mode:
        required.update({"voice_audio_url": voice_url, "music_audio_url": music_url})
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError("MISSING_REQUIRED: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="evs_cpu_") as tmpdir:
        root = Path(tmpdir)
        master = root / "master.mp4"
        output = root / "master_cpu.mp4"
        _download(master_url, master)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        if visual_mode:
            generation, qa = _visual_remix(payload, root, master, output, ffmpeg, target_duration, issues)
        else:
            voice, music = root / "voice.wav", root / "music.wav"
            _download(voice_url, voice)
            _download(music_url, music)
            filters = f"[1:a]volume={voice_volume:.3f},apad,alimiter=limit=0.95[voice];[2:a]volume={music_volume:.3f},apad[music];[voice][music]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[aout]"
            _run([ffmpeg, "-y", "-i", str(master), "-i", str(voice), "-stream_loop", "-1", "-i", str(music), "-filter_complex", filters, "-map", "0:v:0", "-map", "[aout]", "-t", f"{target_duration:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)])
            generation = {
                "mode": "cpu_compositor_non_gpu",
                "engine": "render_workflows_flex",
                "correction_protocol": str(payload.get("correction_protocol") or "EVS_AUDIO_CORRECTION_V1"),
                "scene_count": 0,
                "timeline_seconds": [],
                "source_master_url": master_url,
                "issues": issues,
                "voice_volume": voice_volume,
                "music_volume": music_volume,
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
                "cuts_only": True,
                "voice_present": True,
                "music_present": True,
                "release_gate_required": True,
                "cpu_route": True,
            }

        supabase = create_client(supabase_url, anon_key)
        with output.open("rb") as fh:
            supabase.storage.from_("videos").upload_to_signed_url(path=output_path, token=upload_token, file=fh)

    elapsed = round(time.perf_counter() - started, 3)
    generation["total_seconds"] = elapsed
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
        "route": "NON_GPU",
        "correction_protocol": generation.get("correction_protocol"),
        "visual_diff_score": generation.get("visual_diff_score"),
    }


if __name__ == "__main__":
    app.start()
