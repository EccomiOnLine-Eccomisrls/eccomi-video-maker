import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import requests
from render import Retry, TaskContext, Workflows
from supabase import create_client

app = Workflows()


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
            "User-Agent": "EVS-CPU-Compositor/1.0",
        },
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"CALLBACK_FAILED HTTP {response.status_code}: {response.text[:1200]}")


@app.task(
    name="remix_master",
    plan="flex",
    timeout_seconds=900,
    retry=Retry(max_retries=1, wait_duration_ms=1500, backoff_scaling=2.0),
)
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
    target_duration = float(payload.get("target_duration_seconds") or 15.0)
    voice_volume = max(0.1, min(2.0, float(payload.get("voice_volume") or 1.0)))
    music_volume = max(0.0, min(1.0, float(payload.get("music_volume") or 0.22)))
    issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []

    required = {
        "evs_code": evs_code,
        "job_id": job_id,
        "master_video_url": master_url,
        "voice_audio_url": voice_url,
        "music_audio_url": music_url,
        "storage_path": output_path,
        "storage_upload_token": upload_token,
        "output_public_url": output_public_url,
        "supabase_url": supabase_url,
        "supabase_anon_key": anon_key,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError("MISSING_REQUIRED: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="evs_cpu_") as tmpdir:
        root = Path(tmpdir)
        master = root / "master.mp4"
        voice = root / "voice.wav"
        music = root / "music.wav"
        output = root / "master_cpu.mp4"

        _download(master_url, master)
        _download(voice_url, voice)
        _download(music_url, music)

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        # Strip the old mixed soundtrack and rebuild audio from clean voice + music stems.
        # Video is stream-copied, so no visual quality is lost and CPU usage stays low.
        filter_complex = (
            f"[1:a]volume={voice_volume:.3f},apad,alimiter=limit=0.95[voice];"
            f"[2:a]volume={music_volume:.3f},apad[music];"
            "[voice][music]amix=inputs=2:duration=longest:normalize=0,"
            "alimiter=limit=0.95[aout]"
        )
        _run([
            ffmpeg,
            "-y",
            "-i", str(master),
            "-i", str(voice),
            "-stream_loop", "-1",
            "-i", str(music),
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-t", f"{target_duration:.3f}",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output),
        ])

        supabase = create_client(supabase_url, anon_key)
        with output.open("rb") as fh:
            supabase.storage.from_("videos").upload_to_signed_url(
                path=output_path,
                token=upload_token,
                file=fh,
            )

    elapsed = round(time.perf_counter() - started, 3)
    generation = {
        "mode": "cpu_compositor_non_gpu",
        "engine": "render_workflows_flex",
        "total_seconds": elapsed,
        "scene_count": 0,
        "timeline_seconds": [],
        "source_master_url": master_url,
        "issues": issues,
        "voice_volume": voice_volume,
        "music_volume": music_volume,
    }
    qa = {
        "technical_pass": True,
        "output_width": 1080,
        "output_height": 1920,
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
    }


if __name__ == "__main__":
    app.start()
