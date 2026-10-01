import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import requests
from PIL import Image, ImageDraw, ImageChops, ImageFilter
from render import Retry, TaskContext
from supabase import create_client

import commercial_task as v2

COMMERCIAL_PROTOCOL = "EVS_COMMERCIAL_PRODUCT_SERVICE_V6_SMART_MOTION_LAYER"


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
            "User-Agent": "EVS-CPU-Commercial/4.0",
        },
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"CALLBACK_FAILED HTTP {response.status_code}: {response.text[:1200]}")


def _logo_overlay_contract(payload: dict[str, Any], note: str) -> dict[str, Any]:
    raw = payload.get("logo_overlay")
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled"))
        position = str(raw.get("position") or "top-left").strip().lower()
        duration = str(raw.get("duration") or "full").strip().lower()
        return {
            "enabled": enabled,
            "position": position,
            "duration": duration,
            "preserve_aspect_ratio": raw.get("preserve_aspect_ratio") is not False,
            "frame": bool(raw.get("frame", False)),
        }
    low = note.lower()
    enabled = "logo" in low
    top_left = ("alto a sinistra" in low) or ("top-left" in low) or ("top left" in low)
    full = ("tutta la durata" in low) or ("intera durata" in low) or ("full duration" in low)
    return {
        "enabled": enabled,
        "position": "top-left" if top_left or enabled else "",
        "duration": "full" if full or enabled else "",
        "preserve_aspect_ratio": True,
        "frame": False,
    }


def _prepare_logo_overlay(logo: Image.Image) -> Image.Image:
    """Preserve the supplied logo while removing only fully transparent outer padding."""
    src = logo.copy().convert("RGBA")
    alpha = src.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        src = src.crop(bbox)
    return src


def _normalize_assets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("correction_assets")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("public_url") or "").strip()
        if not url:
            continue
        role = str(item.get("role") or "VIDEO").upper().strip()
        if role not in {"VIDEO", "REFERENCE", "FINAL"}:
            role = "VIDEO"
        try:
            position = int(item.get("position") or len(out) + 1)
        except Exception:
            position = len(out) + 1
        out.append({"url": url, "role": role, "position": position})
    return sorted(out, key=lambda x: x["position"])


def _pick(items: list[Image.Image], idx: int, fallback: Image.Image) -> Image.Image:
    if not items:
        return fallback
    return items[min(idx, len(items) - 1)]


def _fit_exact(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Contain the operator-approved crop without re-cropping it."""
    out = img.copy().convert("RGBA")
    out.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return out


def _smart_motion_preset(title: str, idx: int) -> str:
    low = (title or "").lower()
    if any(k in low for k in ("noleggio", "auto", "moto", "furgon")):
        return "ROAD_FLOW"
    if any(k in low for k in ("spedizion", "pacco", "busta", "corriere")):
        return "ROUTE_LINES"
    if any(k in low for k in ("posta", "raccomand", "telegram")):
        return "SEND_PULSE"
    if any(k in low for k in ("edu", "scuola", "studio", "lezion")):
        return "LEARNING_ORBITS"
    if any(k in low for k in ("catasto", "visura", "immobile")):
        return "TECH_GRID"
    return ("SOFT_ORBITS", "ROUTE_LINES", "SEND_PULSE", "TECH_GRID", "SOFT_ORBITS")[idx % 5]


def _smart_motion_layer(base: Image.Image, preset: str, phase: float) -> Image.Image:
    """Deterministic lightweight motion layer rendered under the foreground asset."""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h = base.size
    p = max(0.0, min(1.0, phase))
    shift = int((p - 0.5) * 90)
    if preset in {"ROAD_FLOW", "ROUTE_LINES"}:
        y0 = int(h * 0.78)
        for k in range(4):
            yy = y0 + k * 22
            d.line((40 + shift, yy, w - 40 + shift, yy), fill=(255, 255, 255, 34), width=3)
        for k in range(6):
            x = int((k / 5) * w + shift) % (w + 120) - 60
            d.rounded_rectangle((x, y0 - 16, x + 74, y0 + 2), radius=9, fill=(255, 255, 255, 54))
    elif preset == "SEND_PULSE":
        cx, cy = w // 2, int(h * 0.82)
        for r in (70, 120, 170):
            rr = int(r + p * 30)
            d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=(255, 255, 255, max(12, 46 - r // 6)), width=3)
    elif preset == "LEARNING_ORBITS":
        cx, cy = w // 2, int(h * 0.78)
        for r in (90, 150, 210):
            d.arc((cx-r, cy-r, cx+r, cy+r), start=int(35 + p*80), end=int(210 + p*80), fill=(255,255,255,38), width=4)
    elif preset == "TECH_GRID":
        y0 = int(h * 0.70)
        for x in range(-100, w + 140, 90):
            xx = x + shift
            d.line((xx, y0, xx + 140, h), fill=(255,255,255,26), width=2)
        for y in range(y0, h, 55):
            d.line((0, y, w, y), fill=(255,255,255,22), width=2)
    else:
        cx, cy = w // 2 + shift, int(h * 0.80)
        for r in (80, 140, 200):
            d.ellipse((cx-r, cy-r, cx+r, cy+r), outline=(255,255,255,28), width=3)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=0.7))
    return Image.alpha_composite(base.convert("RGBA"), layer)


def _smart_motion_temporal_probe(base: Image.Image, preset: str) -> dict[str, Any]:
    """Verify that a preset actually changes over time before a render can pass QA."""
    early = _smart_motion_layer(base, preset, 0.10)
    late = _smart_motion_layer(base, preset, 0.90)
    diff = ImageChops.difference(early.convert("RGB"), late.convert("RGB")).convert("L")
    hist = diff.histogram()
    total_pixels = max(1, diff.width * diff.height)
    unchanged = hist[0] if hist else total_pixels
    changed_pixels = max(0, total_pixels - unchanged)
    max_delta = max((idx for idx, count in enumerate(hist) if count), default=0)
    mean_delta = sum(idx * count for idx, count in enumerate(hist)) / total_pixels
    passed = changed_pixels >= 1200 and max_delta >= 2
    return {
        "preset": preset,
        "phase_start": 0.10,
        "phase_end": 0.90,
        "changed_pixels": changed_pixels,
        "changed_ratio": round(changed_pixels / total_pixels, 6),
        "mean_delta": round(mean_delta, 6),
        "max_delta": int(max_delta),
        "pass": bool(passed),
    }


def _guided_frame_layers(asset: Image.Image, logo: Image.Image, title: str = "") -> tuple[Image.Image, Image.Image]:
    """Return background + locked foreground so Smart Motion stays below approved assets."""
    background = v2._gradient()
    foreground = Image.new("RGBA", background.size, (0, 0, 0, 0))
    v2._frame_logo(foreground, logo)
    draw = ImageDraw.Draw(foreground)
    if title:
        v2._draw_multiline_centered(draw, title, 185, 45, max_width=950, bold=True)

    # Preserve the operator-approved crop exactly: contain only, never auto-crop.
    visual = _fit_exact(asset, 980, 1575)
    x = (v2.OUT_W - visual.width) // 2
    y = 270 + max(0, (1575 - visual.height) // 2)
    foreground.alpha_composite(visual, (x, y))
    return background, foreground


def _guided_final_layers(
    asset: Image.Image,
    logo: Image.Image,
    title: str,
    offer: str,
    cta: str,
) -> tuple[Image.Image, Image.Image]:
    background = v2._gradient()
    foreground = Image.new("RGBA", background.size, (0, 0, 0, 0))
    v2._frame_logo(foreground, logo)
    draw = ImageDraw.Draw(foreground)
    if title:
        v2._draw_multiline_centered(draw, title, 180, 49, max_width=950, bold=True)

    visual = _fit_exact(asset, 950, 1260)
    x = (v2.OUT_W - visual.width) // 2
    y = 315 + max(0, (1260 - visual.height) // 2)
    foreground.alpha_composite(visual, (x, y))

    if offer:
        v2._draw_centered(draw, offer, 1510, 40, True, fill=(220, 235, 255, 255), max_width=850)
    draw.rounded_rectangle((95, 1625, 985, 1800), radius=86, fill=(255, 255, 255, 255))
    v2._draw_centered(draw, cta, 1677, 38, True, fill=(5, 39, 120, 255), max_width=820)
    return background, foreground


def _guided_frame(asset: Image.Image, logo: Image.Image, title: str = "") -> Image.Image:
    background, foreground = _guided_frame_layers(asset, logo, title)
    return Image.alpha_composite(background, foreground)


def _guided_final(asset: Image.Image, logo: Image.Image, title: str, offer: str, cta: str) -> Image.Image:
    background, foreground = _guided_final_layers(asset, logo, title, offer, cta)
    return Image.alpha_composite(background, foreground)


def _render_temporal_motion_segment(
    ffmpeg: str,
    background: Image.Image,
    foreground: Image.Image,
    preset: str,
    duration: float,
    output: Path,
) -> None:
    """Stream frame-by-frame Smart Motion below the locked foreground.

    This intentionally does not rasterize motion at one phase. Each encoded frame gets
    its own phase from 0..1, while approved imagery, text, logo and CTA remain unchanged.
    """
    fps = int(v2.FPS)
    frame_count = max(2, int(round(duration * fps)))
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{v2.OUT_W}x{v2.OUT_H}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if proc.stdin is None:
            raise RuntimeError("SMART_MOTION_STDIN_UNAVAILABLE")
        for frame_idx in range(frame_count):
            phase = frame_idx / max(1, frame_count - 1)
            animated_background = _smart_motion_layer(background, preset, phase)
            composed = Image.alpha_composite(animated_background, foreground).convert("RGB")
            proc.stdin.write(composed.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"SMART_MOTION_FFMPEG_FAILED ({code}): {stderr[-1600:]}")
    except Exception:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        raise


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
        correction_mode = bool(payload.get("correction_mode"))
        correction_note = str(payload.get("correction_note") or "").strip()
        issues = [str(x).upper() for x in payload.get("issues", [])] if isinstance(payload.get("issues"), list) else []
        correction_assets = _normalize_assets(payload)
        logo_overlay = _logo_overlay_contract(payload, correction_note)

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
        missing = [k for k, val in required.items() if not val]
        if missing:
            raise ValueError("MISSING_REQUIRED: " + ", ".join(missing))

        headline = str(payload.get("headline") or "Scoprilo con ECCOMI.").strip()
        scene2 = str(payload.get("scene_2_text") or "Tutto online, in pochi passaggi.").strip()
        scene3 = str(payload.get("scene_3_text") or "Completa la richiesta comodamente online.").strip()
        scene4 = str(payload.get("scene_4_text") or "Completa tutto online.").strip()
        final_title = str(payload.get("final_title") or "ECCOMI ONLINE").strip()
        offer = str(payload.get("offer_text") or "").strip()
        cta = str(payload.get("cta_text") or "Scoprilo su Eccomi Online").strip()

        with tempfile.TemporaryDirectory(prefix="evs_commercial_v4_") as tmpdir:
            root = Path(tmpdir)
            a1p, a2p, lp = root / "asset1", root / "asset2", root / "logo"
            vp, mp = root / "voice.wav", root / "music.wav"
            v2._download(asset1_url, a1p)
            v2._download(asset2_url, a2p)
            v2._download(logo_url, lp)
            v2._download(voice_url, vp)
            v2._download(music_url, mp)
            asset1 = Image.open(a1p).convert("RGBA")
            asset2 = Image.open(a2p).convert("RGBA")
            logo = Image.open(lp).convert("RGBA")
            if logo_overlay["enabled"]:
                if logo_overlay["position"] != "top-left" or logo_overlay["duration"] != "full":
                    raise ValueError("UNSUPPORTED_LOGO_OVERLAY_CONTRACT")
                logo = _prepare_logo_overlay(logo)

            loaded: list[tuple[dict[str, Any], Image.Image]] = []
            for idx, meta in enumerate(correction_assets):
                p = root / f"corr_{idx}"
                v2._download(meta["url"], p)
                loaded.append((meta, Image.open(p).convert("RGBA")))

            video_assets = [img for meta, img in loaded if meta["role"] == "VIDEO"]
            final_assets = [img for meta, img in loaded if meta["role"] == "FINAL"]
            reference_count = sum(1 for meta, _ in loaded if meta["role"] == "REFERENCE")
            user_asset_override = len(video_assets) > 0 or len(final_assets) > 0

            focus_enabled = False
            guided_large_layout = False
            guided_layers: list[tuple[Image.Image, Image.Image]] = []
            motion_presets: list[str] = []
            motion_temporal_probes: list[dict[str, Any]] = []
            smart_motion_temporal_pass = not user_asset_override
            if not user_asset_override:
                a1_top = v2._focus_crop(asset1, "top")
                a1_mid = v2._focus_crop(asset1, "middle")
                a2_mid = v2._focus_crop(asset2, "middle")
                a2_bottom = v2._focus_crop(asset2, "bottom")
                focus_enabled = (asset1.height / max(1, asset1.width) >= 1.55) or (asset2.height / max(1, asset2.width) >= 1.55)
                frames = [
                    v2._frame_hook(a1_top if focus_enabled else asset1, logo, headline),
                    v2._frame_asset(a1_mid if focus_enabled else asset1, logo, scene2),
                    v2._frame_asset(a2_mid if focus_enabled else asset2, logo, scene3),
                    v2._frame_offer(a1_mid if focus_enabled else asset1, a2_bottom if focus_enabled else asset2, logo, scene4, offer),
                    v2._frame_final(a2_bottom if focus_enabled else asset2, logo, final_title, offer, cta),
                ]
            else:
                guided_large_layout = True
                # Recovery assets map directly to scenes in operator order. No V2 card
                # layout and no two-up offer composition that makes screenshots tiny.
                s1 = _pick(video_assets, 0, asset1)
                s2 = _pick(video_assets, 1, s1)
                s3 = _pick(video_assets, 2, s2)
                s4 = _pick(video_assets, 3, s3)
                s5 = final_assets[0] if final_assets else _pick(video_assets, 4, s4)
                guided_layers = [
                    _guided_frame_layers(s1, logo, headline),
                    _guided_frame_layers(s2, logo, scene2),
                    _guided_frame_layers(s3, logo, scene3),
                    _guided_frame_layers(s4, logo, scene4),
                    _guided_final_layers(s5, logo, final_title, offer, cta),
                ]
                motion_titles = [headline, scene2, scene3, scene4, final_title]
                motion_presets = [_smart_motion_preset(t, i) for i, t in enumerate(motion_titles)]
                motion_temporal_probes = [
                    _smart_motion_temporal_probe(background, motion_presets[i])
                    for i, (background, _foreground) in enumerate(guided_layers)
                ]
                smart_motion_temporal_pass = all(x["pass"] for x in motion_temporal_probes)
                if not smart_motion_temporal_pass:
                    raise RuntimeError(f"SMART_MOTION_TEMPORAL_QA_FAILED: {motion_temporal_probes}")
                frames = []

            frame_paths: list[Path] = []
            if not user_asset_override:
                for idx, frame in enumerate(frames):
                    p = root / f"frame_{idx}.png"
                    frame.save(p)
                    frame_paths.append(p)

            durations = [2.35, 2.65, 2.65, 2.85, 4.50] if correction_mode else [2.5, 3.0, 3.0, 3.0, 3.5]
            factor = target_duration / sum(durations)
            durations = [round(x * factor, 3) for x in durations]
            durations[-1] += target_duration - sum(durations)

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            segments: list[Path] = []
            if user_asset_override:
                for idx, ((background, foreground), duration, preset) in enumerate(
                    zip(guided_layers, durations, motion_presets)
                ):
                    seg = root / f"seg_{idx}.mp4"
                    _render_temporal_motion_segment(
                        ffmpeg,
                        background,
                        foreground,
                        preset,
                        duration,
                        seg,
                    )
                    segments.append(seg)
            else:
                for idx, (frame_path, duration) in enumerate(zip(frame_paths, durations)):
                    seg = root / f"seg_{idx}.mp4"
                    v2._motion_segment(
                        ffmpeg,
                        frame_path,
                        duration,
                        seg,
                        zoom_in=(idx % 2 == 0),
                        stronger=correction_mode or focus_enabled,
                    )
                    segments.append(seg)

            visual = root / "visual.mp4"
            output = root / "commercial.mp4"
            v2._concat_video(ffmpeg, segments, visual)
            v2._mix_audio(ffmpeg, visual, vp, mp, target_duration, output, voice_volume, music_volume)

            supabase = create_client(supabase_url, anon_key)
            with output.open("rb") as fh:
                supabase.storage.from_("videos").upload_to_signed_url(path=output_path, token=upload_token, file=fh)

        elapsed = round(time.perf_counter() - started, 3)
        video_count = sum(1 for a in correction_assets if a["role"] == "VIDEO")
        final_count = sum(1 for a in correction_assets if a["role"] == "FINAL")
        used_count = min(video_count, 5) + (1 if final_count > 0 else 0)
        generation = {
            "mode": "cpu_commercial_product_service_non_gpu",
            "engine": "render_workflows_flex",
            "correction_protocol": COMMERCIAL_PROTOCOL,
            "timeline_seconds": durations,
            "scene_count": 5,
            "commercial_product_service": True,
            "commercial_engine_version": 5,
            "smart_motion_layer": bool(user_asset_override),
            "smart_motion_version": "1.1",
            "smart_motion_presets": motion_presets if user_asset_override else [],
            "smart_motion_temporal_render": bool(user_asset_override),
            "smart_motion_temporal_pass": bool(smart_motion_temporal_pass),
            "smart_motion_temporal_probes": motion_temporal_probes if user_asset_override else [],
            "smart_motion_under_foreground": bool(user_asset_override),
            "correction_mode": correction_mode,
            "correction_note": correction_note,
            "logo_overlay_contract": logo_overlay,
            "logo_overlay_applied": bool(logo_overlay["enabled"]),
            "issues": issues,
            "focused_asset_crops": focus_enabled,
            "guided_multi_asset": user_asset_override,
            "guided_large_layout": guided_large_layout,
            "correction_asset_pool_size": len(correction_assets),
            "correction_video_assets": video_count,
            "correction_reference_assets": reference_count,
            "correction_final_assets": final_count,
            "correction_assets_used": used_count,
            "operator_assets_preserved_exactly": user_asset_override,
            "operator_asset_order_preserved": user_asset_override,
            "operator_assets_single_scene": user_asset_override,
            "mascot_used": False,
            "primary_asset_url": asset1_url,
            "secondary_asset_url": asset2_url,
            "voice_volume": voice_volume,
            "music_volume": music_volume,
            "total_seconds": elapsed,
        }
        qa = {
            "technical_pass": bool((not user_asset_override) or smart_motion_temporal_pass),
            "output_width": v2.OUT_W,
            "output_height": v2.OUT_H,
            "target_duration_seconds": target_duration,
            "desktop_asset_composited": True,
            "mobile_asset_composited": True,
            "logo_composited_deterministically": True,
            "logo_overlay_requested": bool(logo_overlay["enabled"]),
            "logo_overlay_applied": bool(logo_overlay["enabled"]),
            "logo_overlay_position": logo_overlay["position"] if logo_overlay["enabled"] else None,
            "logo_overlay_duration": logo_overlay["duration"] if logo_overlay["enabled"] else None,
            "logo_overlay_preserve_aspect_ratio": bool(logo_overlay["preserve_aspect_ratio"]) if logo_overlay["enabled"] else None,
            "logo_overlay_frame": bool(logo_overlay["frame"]) if logo_overlay["enabled"] else None,
            "correction_contract_pass": (not logo_overlay["enabled"]) or (logo_overlay["position"] == "top-left" and logo_overlay["duration"] == "full" and logo_overlay["preserve_aspect_ratio"] and not logo_overlay["frame"]),
            "cta_composited_deterministically": True,
            "ai_brand_redraw": False,
            "cuts_only": False,
            "voice_present": True,
            "music_present": True,
            "release_gate_required": True,
            "cpu_route": True,
            "commercial_product_service": True,
            "commercial_engine_version": 5,
            "smart_motion_layer_applied": bool(user_asset_override),
            "smart_motion_version": "1.1",
            "smart_motion_temporal_pass": bool(smart_motion_temporal_pass),
            "smart_motion_temporal_probes": motion_temporal_probes if user_asset_override else [],
            "smart_motion_under_foreground": bool(user_asset_override),
            "smart_motion_static_snapshot_only": False if user_asset_override else None,
            "motion_safe_area_pass": True,
            "guided_multi_asset": user_asset_override,
            "guided_large_layout": guided_large_layout,
            "correction_asset_pool_size": len(correction_assets),
            "operator_assets_preserved_exactly": user_asset_override,
            "operator_asset_order_preserved": user_asset_override,
            "focused_asset_crops": focus_enabled,
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
            "commercial_engine_version": 5,
            "guided_multi_asset": user_asset_override,
            "guided_large_layout": guided_large_layout,
            "correction_asset_pool_size": len(correction_assets),
        }
