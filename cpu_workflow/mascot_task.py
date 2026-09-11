import tempfile
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import requests
from PIL import Image, ImageDraw
from render import Retry, TaskContext
from supabase import create_client

import commercial_task as base

MASCOT_PROTOCOL = "EVS_MASCOT_FINAL_V3_CONTINUOUS"
OUT_W, OUT_H, FPS = base.OUT_W, base.OUT_H, base.FPS


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
            "User-Agent": "EVS-CPU-Mascot/3.0",
        },
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"CALLBACK_FAILED HTTP {response.status_code}: {response.text[:1200]}")


def _frame(mascot: Image.Image, logo: Image.Image | None, title: str, subtitle: str = "", cta: str = "") -> Image.Image:
    canvas = base._gradient()
    draw = ImageDraw.Draw(canvas)
    if logo is not None:
        base._frame_logo(canvas, logo)
    if title:
        base._draw_multiline_centered(draw, title, 175, 56, max_width=940, bold=True)
    visual = base._fit(mascot, 800, 1120)
    canvas.alpha_composite(visual, ((OUT_W - visual.width) // 2, 430 + max(0, (1040 - visual.height) // 2)))
    if subtitle:
        base._draw_multiline_centered(draw, subtitle, 1505, 34, max_width=900, fill=(220, 235, 255, 255), bold=False)
    if cta:
        draw.rounded_rectangle((100, 1650, 980, 1810), radius=80, fill=(255, 255, 255, 255))
        base._draw_centered(draw, cta, 1698, 38, True, fill=(5, 39, 120, 255), max_width=810)
    return canvas


def _continuous_approved_video(ffmpeg: str, raw_video: Path, duration: float, output: Path) -> None:
    """Use the approved generated video continuously for the full master.

    Grok already returns a portrait 9:16 asset. This stage may normalize it to
    EVS output dimensions, but it must never alternate it with static mascot cards
    or restart the source at every scene boundary.
    """
    filt = (
        f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=0x061d55,"
        f"fps={FPS},format=yuv420p[v]"
    )
    base._run([
        ffmpeg, "-y", "-stream_loop", "-1", "-i", str(raw_video),
        "-filter_complex", filt, "-map", "[v]", "-t", f"{duration:.3f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-movflags", "+faststart", str(output),
    ])


def _mix_optional_audio(ffmpeg: str, visual: Path, voice: Path | None, music: Path | None, target: float, output: Path, voice_volume: float, music_volume: float) -> None:
    if voice is not None and music is not None:
        base._mix_audio(ffmpeg, visual, voice, music, target, output, voice_volume, music_volume)
        return
    if voice is not None:
        base._run([
            ffmpeg, "-y", "-i", str(visual), "-i", str(voice),
            "-filter_complex", f"[1:a]volume={voice_volume:.3f},apad,alimiter=limit=0.92[a]",
            "-map", "0:v:0", "-map", "[a]", "-t", f"{target:.3f}",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ])
        return
    if music is not None:
        base._run([
            ffmpeg, "-y", "-i", str(visual), "-stream_loop", "-1", "-i", str(music),
            "-filter_complex", f"[1:a]volume={music_volume:.3f},apad,alimiter=limit=0.92[a]",
            "-map", "0:v:0", "-map", "[a]", "-t", f"{target:.3f}",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ])
        return
    base._run([ffmpeg, "-y", "-i", str(visual), "-t", f"{target:.3f}", "-c", "copy", str(output)])


def register_mascot(app) -> None:
    @app.task(name="create_mascot_final", plan="flex", timeout_seconds=900, retry=Retry(max_retries=1, wait_duration_ms=1500, backoff_scaling=2.0))
    def create_mascot_final(ctx: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        evs_code = str(payload.get("evs_code") or "").strip()
        job_id = str(payload.get("job_id") or "").strip()
        raw_video_url = str(payload.get("raw_gpu_video_url") or "").strip()
        mascot_url = str(payload.get("mascot_image_url") or "").strip()
        logo_url = str(payload.get("logo_url") or "").strip()
        voice_url = str(payload.get("voice_audio_url") or "").strip()
        music_url = str(payload.get("music_audio_url") or "").strip()
        output_path = str(payload.get("storage_path") or "").strip()
        upload_token = str(payload.get("storage_upload_token") or "").strip()
        output_public_url = str(payload.get("output_public_url") or "").strip()
        supabase_url = str(payload.get("supabase_url") or "").strip()
        anon_key = str(payload.get("supabase_anon_key") or "").strip()
        target = max(8.0, min(30.0, float(payload.get("target_duration_seconds") or 15.0)))
        voice_volume = max(0.1, min(2.0, float(payload.get("voice_volume") or 1.0)))
        music_volume = max(0.0, min(1.0, float(payload.get("music_volume") or 0.20)))
        gpu_identity_approved = payload.get("gpu_identity_approved") is True
        identity_gate_status = str(payload.get("identity_gate_status") or ("PASS" if gpu_identity_approved else "PENDING")).strip().upper()

        required = {
            "evs_code": evs_code,
            "job_id": job_id,
            "raw_gpu_video_url": raw_video_url,
            "mascot_image_url": mascot_url,
            "storage_path": output_path,
            "storage_upload_token": upload_token,
            "output_public_url": output_public_url,
            "supabase_url": supabase_url,
            "supabase_anon_key": anon_key,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError("MISSING_REQUIRED: " + ", ".join(missing))

        brand = str(payload.get("brand_name") or "ECCOMI ONLINE").strip()
        headline = str(payload.get("headline") or "La tua mascotte. Il tuo brand.").strip()
        message = str(payload.get("message") or "Una presenza riconoscibile, coerente e sempre tua.").strip()
        cta = str(payload.get("cta_text") or "Scoprilo con Eccomi Online").strip()
        mascot_name = str(payload.get("mascot_name") or "MASCOTTE AI").strip()

        with tempfile.TemporaryDirectory(prefix="evs_mascot_final_") as tmpdir:
            root = Path(tmpdir)
            rawp, mascp = root / "provider.mp4", root / "mascot"
            voicep, musicp, logop = root / "voice.wav", root / "music.wav", root / "logo"
            base._download(raw_video_url, rawp)
            base._download(mascot_url, mascp)
            if voice_url:
                base._download(voice_url, voicep)
            if music_url:
                base._download(music_url, musicp)
            if logo_url:
                base._download(logo_url, logop)

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            visual = root / "visual.mp4"
            output = root / "mascot_final.mp4"

            if gpu_identity_approved:
                _continuous_approved_video(ffmpeg, rawp, target, visual)
                layout_mode = "CONTINUOUS_APPROVED_PROVIDER_VIDEO"
                scene_count = 1
                timeline = [target]
            else:
                mascot = Image.open(mascp).convert("RGBA")
                logo = Image.open(logop).convert("RGBA") if logo_url else None
                fallback = _frame(mascot, logo, headline or brand, mascot_name, cta)
                frame_path = root / "fallback.png"
                fallback.save(frame_path)
                base._motion_segment(ffmpeg, frame_path, target, visual, zoom_in=True, stronger=False)
                layout_mode = "OFFICIAL_ASSET_SAFE_STATIC"
                scene_count = 1
                timeline = [target]

            _mix_optional_audio(
                ffmpeg,
                visual,
                voicep if voice_url else None,
                musicp if music_url else None,
                target,
                output,
                voice_volume,
                music_volume,
            )

            supabase = create_client(supabase_url, anon_key)
            with output.open("rb") as fh:
                supabase.storage.from_("videos").upload_to_signed_url(path=output_path, token=upload_token, file=fh)

        elapsed = round(time.perf_counter() - started, 3)
        generation = {
            "mode": "cpu_mascot_final_v3_continuous",
            "engine": "render_workflows_flex",
            "correction_protocol": MASCOT_PROTOCOL,
            "width": OUT_W,
            "height": OUT_H,
            "fps": FPS,
            "frames": int(round(target * FPS)),
            "output_duration_seconds": target,
            "timeline_seconds": timeline,
            "scene_count": scene_count,
            "raw_gpu_video_url": raw_video_url,
            "mascot_image_url": mascot_url,
            "voice_volume": voice_volume,
            "music_volume": music_volume,
            "identity_gate_status": identity_gate_status,
            "gpu_identity_approved": gpu_identity_approved,
            "layout_mode": layout_mode,
            "text_overlay_deferred": True,
            "message": message,
            "cta_text": cta,
            "total_seconds": elapsed,
        }
        qa = {
            "technical_pass": True,
            "output_width": OUT_W,
            "output_height": OUT_H,
            "target_duration_seconds": target,
            "mascot_asset_composited": not gpu_identity_approved,
            "gpu_animation_composited": gpu_identity_approved,
            "identity_gate_required": True,
            "identity_gate_passed": gpu_identity_approved,
            "identity_gate_status": identity_gate_status,
            "voice_present": bool(voice_url),
            "music_present": bool(music_url),
            "release_gate_required": True,
            "ai_brand_redraw": False,
            "cpu_mascot_final": True,
            "continuous_source_preserved": gpu_identity_approved,
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
            "route": "MASCOT_PROVIDER_PLUS_CPU_FINAL_V3_CONTINUOUS",
            "identity_gate_status": identity_gate_status,
            "gpu_identity_approved": gpu_identity_approved,
            "correction_protocol": MASCOT_PROTOCOL,
        }
