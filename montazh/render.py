"""Рендер: нарезка сегментов, сборка, звук.

Почему сегментами, а не одной мега-командой: так видны ошибки на конкретном
плане, можно перезапускать только упавшие куски и грузить все ядра. Каждый
сегмент рендерится ровно в N кадров, поэтому склейки не плывут относительно
музыки.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .env import Tools, run
from .plan import Plan, Segment

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXT


def segment_filter(plan: Plan, seg: Segment) -> str:
    w, h = plan.width, plan.height
    parts = [
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={w}:{h}",
    ]
    if seg.hflip:
        parts.append("hflip")
    if abs(seg.zoom_to - seg.zoom_from) > 1e-4:
        span = max(seg.frames - 1, 1)
        step = (seg.zoom_to - seg.zoom_from) / span
        parts.append(
            "zoompan=z='{:.5f}+{:.6f}*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            "d=1:s={}x{}:fps={}".format(seg.zoom_from, step, w, h, plan.fps)
        )
    if abs(seg.speed - 1.0) > 1e-4:
        parts.append(f"setpts=PTS/{seg.speed:.5f}")
    parts.append(f"fps={plan.fps}")
    # Если шот короче слота, добираем кадры клоном последнего — лучше, чем дырка.
    expected = seg.src_dur / max(seg.speed, 1e-6)
    need = seg.render_frames / plan.fps
    pad = max(0.0, need - expected + 0.08)
    if pad > 0.02:
        parts.append(f"tpad=stop_mode=clone:stop_duration={pad:.3f}")
    parts.append("setsar=1")
    parts.append("format=yuv420p")
    return ",".join(parts)


def plan_key(plan: Plan) -> str:
    """Отпечаток плана: сегменты от разных планов не должны перемешиваться в кеше."""
    import hashlib

    h = hashlib.sha1()
    h.update(f"{plan.width}x{plan.height}@{plan.fps}".encode())
    for s in plan.segments:
        h.update(
            f"|{s.src}:{s.src_in}:{s.src_dur}:{s.speed}:{s.frames}:"
            f"{s.zoom_from}:{s.zoom_to}:{s.hflip}:{s.transition}:{s.trans_dur}".encode()
        )
    return h.hexdigest()[:10]


def segment_cmd(tools: Tools, plan: Plan, seg: Segment, out_path: str) -> list[str]:
    cmd = [tools.ffmpeg, "-y", "-v", "error", "-nostdin"]
    if _is_image(seg.src):
        cmd += ["-loop", "1", "-framerate", str(plan.fps), "-t", f"{seg.src_dur:.3f}", "-i", seg.src]
    else:
        cmd += ["-ss", f"{seg.src_in:.3f}", "-t", f"{seg.src_dur:.3f}", "-i", seg.src]
    cmd += [
        "-vf", segment_filter(plan, seg),
        "-frames:v", str(seg.render_frames),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-g", str(plan.fps),
        "-r", str(plan.fps),
        out_path,
    ]
    return cmd


def render_segments(plan: Plan, tools: Tools, workdir: Path, workers: int | None = None,
                    on_progress=None) -> list[str]:
    seg_dir = workdir / f"segments_{plan_key(plan)}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(2, min(8, (os.cpu_count() or 4) - 1))
    paths = [str(seg_dir / f"seg_{i:04d}.mp4") for i in range(len(plan.segments))]

    def job(idx: int) -> tuple[int, str]:
        seg = plan.segments[idx]
        out = paths[idx]
        if os.path.exists(out) and os.path.getsize(out) > 1024:
            return idx, out
        cmd = segment_cmd(tools, plan, seg, out)
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Сегмент {idx} не отрендерился:\n{proc.stderr[-1500:]}"
            )
        return idx, out

    done = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, i): i for i in range(len(plan.segments))}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
            done += 1
            if on_progress and (done % 10 == 0 or done == len(plan.segments)):
                on_progress(done, len(plan.segments))
    if errors:
        raise RuntimeError("\n".join(errors[:3]))
    return paths


def _durations(plan: Plan) -> list[float]:
    return [s.frames / plan.fps for s in plan.segments]


def assemble(plan: Plan, tools: Tools, segments: list[str], workdir: Path) -> str:
    """Склейка: резкие каты — через concat, наплывы — через цепочку xfade."""
    out = str(workdir / "video.mp4")
    has_fade = any(s.transition != "cut" and s.trans_dur > 0 for s in plan.segments[:-1])

    if not has_fade:
        listing = workdir / "concat.txt"
        listing.write_text(
            "\n".join(f"file '{Path(p).resolve().as_posix()}'" for p in segments), "utf-8"
        )
        run([
            tools.ffmpeg, "-y", "-v", "error", "-nostdin",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", "-movflags", "+faststart", out,
        ])
        return out

    durs = _durations(plan)
    inputs: list[str] = []
    for path in segments:
        inputs += ["-i", path]

    # У каждого сегмента отрезаем ровно его длину (плановая + хвост под переход).
    chain: list[str] = []
    for i, seg in enumerate(plan.segments):
        length = seg.render_frames / plan.fps
        chain.append(f"[{i}:v]trim=start=0:end={length:.4f},setpts=PTS-STARTPTS[t{i}]")

    prev = "[t0]"
    total = durs[0]
    for i in range(1, len(segments)):
        prev_seg = plan.segments[i - 1]
        if prev_seg.transition == "cut" or prev_seg.trans_dur <= 0:
            label = f"[c{i}]"
            chain.append(f"{prev}[t{i}]concat=n=2:v=1:a=0{label}")
            total += durs[i]
        else:
            d = prev_seg.trans_dur
            # Наплыв начинается точно на плановой склейке: у сегмента есть хвост
            # ровно на d, поэтому длительность остаётся равной плану.
            offset = total
            label = f"[x{i}]"
            kind = prev_seg.transition if prev_seg.transition != "cut" else "fade"
            chain.append(
                f"{prev}[t{i}]xfade=transition={kind}:duration={d:.3f}:offset={offset:.3f}{label}"
            )
            total = total + durs[i]
        prev = label

    filt = ";".join(chain)
    run([
        tools.ffmpeg, "-y", "-v", "error", "-nostdin", *inputs,
        "-filter_complex", filt, "-map", prev,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
        "-pix_fmt", "yuv420p", "-r", str(plan.fps), "-movflags", "+faststart", out,
    ])
    return out


def add_audio(plan: Plan, tools: Tools, video: str, out_path: str,
              fade_in: float = 0.25, fade_out: float = 1.6) -> str:
    dur = plan.total_frames / plan.fps
    fade_out = min(fade_out, max(0.2, dur * 0.15))
    audio_filter = (
        "aformat=sample_rates=48000:channel_layouts=stereo,"
        "loudnorm=I=-14:TP=-1.5:LRA=11,"
        f"afade=t=in:st=0:d={fade_in:.2f},"
        f"afade=t=out:st={max(0.0, dur - fade_out):.2f}:d={fade_out:.2f}"
    )
    run([
        tools.ffmpeg, "-y", "-v", "error", "-nostdin",
        "-i", video,
        "-ss", f"{plan.music_start:.3f}", "-t", f"{dur:.3f}", "-i", plan.music_path,
        "-filter_complex", f"[1:a]{audio_filter}[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest", "-movflags", "+faststart",
        out_path,
    ])
    return out_path


def poster(plan: Plan, tools: Tools, video: str, out_path: str, at: float | None = None) -> str:
    """Постер: без явного времени берём самый светлый из нескольких кадров."""
    if at is not None:
        run([
            tools.ffmpeg, "-y", "-v", "error", "-nostdin",
            "-ss", f"{at:.2f}", "-i", video, "-frames:v", "1", "-q:v", "3", out_path,
        ])
        return out_path

    dur = plan.total_frames / plan.fps
    count = 6
    tmp_dir = Path(out_path).parent / "_poster_candidates"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for t in [(i + 0.5) * dur / count for i in range(count)]:
        run([
            tools.ffmpeg, "-y", "-v", "error", "-nostdin",
            "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", "-q:v", "3",
            str(tmp_dir / f"cand_{t:08.2f}.jpg"),
        ])
    best, best_luma = None, -1.0
    try:
        from PIL import Image, ImageStat

        for cand in sorted(tmp_dir.glob("cand_*.jpg")):
            with Image.open(cand) as img:
                luma = float(ImageStat.Stat(img.convert("L")).mean[0])
            if luma > best_luma:
                best, best_luma = cand, luma
    except Exception:
        best = next(iter(sorted(tmp_dir.glob("cand_*.jpg"))), None)
    if best is None:
        raise RuntimeError("Не удалось выбрать кадр для постера")
    Path(out_path).write_bytes(best.read_bytes())
    for cand in tmp_dir.glob("cand_*.jpg"):
        cand.unlink(missing_ok=True)
    return out_path


def contact_sheet(plan: Plan, tools: Tools, video: str, out_path: str,
                  every: float | None = None, cols: int = 6, rows: int = 4) -> str:
    if every is None:
        every = max(0.5, plan.duration / (cols * rows))
    run([
        tools.ffmpeg, "-y", "-v", "error", "-nostdin", "-i", video,
        "-vf", f"fps=1/{every},scale=240:-2,tile={cols}x{rows}",
        "-frames:v", "1", "-q:v", "4", out_path,
    ])
    return out_path


def render(plan: Plan, tools: Tools, out_path: str, workdir: Path,
           workers: int | None = None, on_progress=None) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    key = plan_key(plan)
    run_dir = workdir / f"run_{key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _cleanup(workdir, keep=key)
    segments = render_segments(plan, tools, run_dir, workers=workers, on_progress=on_progress)
    video = assemble(plan, tools, segments, run_dir)
    _verify_frames(plan, tools, video)
    final = add_audio(plan, tools, video, out_path)
    return {"output": final, "video": video, "segments": segments, "workdir": str(run_dir)}


def _cleanup(workdir: Path, keep: str) -> None:
    """Убирает сегменты прошлых планов, чтобы не копить гигабайты."""
    import shutil

    for child in workdir.iterdir():
        if child.is_dir() and (child.name.startswith("segments_") or child.name.startswith("run_")):
            if keep not in child.name:
                shutil.rmtree(child, ignore_errors=True)


def _verify_frames(plan: Plan, tools: Tools, video: str) -> None:
    """Проверяет, что собранное видео совпадает с планом по числу кадров."""
    import json

    proc = run([
        tools.ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", video,
    ])
    try:
        data = json.loads(proc.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        frames = int(stream.get("nb_read_frames") or 0)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except Exception:
        return
    expected = plan.total_frames
    if frames and abs(frames - expected) > 2:
        raise RuntimeError(
            f"Собранное видео разошлось с планом: {frames} кадров вместо {expected}. "
            f"Скорее всего, часть сегментов не отрендерилась."
        )
    if width and (width != plan.width or height != plan.height):
        raise RuntimeError(
            f"Размер видео {width}×{height} не совпадает с планом {plan.width}×{plan.height}"
        )
