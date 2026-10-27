"""Разбор медиафайлов: длительность, разрешение, звук, поворот."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .env import Tools, run

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv", ".flv"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Media:
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    vcodec: str = ""
    acodec: str = ""
    has_audio: bool = False
    rotation: int = 0
    bitrate: int = 0
    kind: str = "video"  # video | audio | image
    raw: dict = field(default_factory=dict)

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 0.0

    def summary(self) -> str:
        if self.kind == "audio":
            return f"{Path(self.path).name} — звук {self.duration:.1f} c, {self.acodec}"
        if self.kind == "image":
            return f"{Path(self.path).name} — фото {self.width}×{self.height}"
        return (
            f"{Path(self.path).name} — {self.width}×{self.height}, "
            f"{self.fps:.2f} fps, {self.duration:.1f} c, {self.vcodec}"
        )


def _num(value, default: float = 0.0) -> float:
    """ffprobe отдаёт "N/A" для картинок и битых файлов — не падаем."""
    try:
        if value in (None, "", "N/A"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_fps(rate: str) -> float:
    try:
        if "/" in rate:
            num, den = rate.split("/")
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(rate)
    except Exception:
        return 0.0


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                return int(float(sd["rotation"])) % 360
            except Exception:
                pass
    try:
        return int(float(stream.get("tags", {}).get("rotate", 0))) % 360
    except Exception:
        return 0


def probe(tools: Tools, path: str) -> Media:
    proc = run(
        [
            tools.ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(proc.stdout or "{}")
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    media = Media(path=str(path), raw=data)
    media.duration = _num(fmt.get("duration")) or _num((video or {}).get("duration"))
    media.bitrate = int(_num(fmt.get("bit_rate")))

    if video:
        media.width = int(video.get("width") or 0)
        media.height = int(video.get("height") or 0)
        media.fps = _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0")
        media.vcodec = video.get("codec_name") or ""
        media.rotation = _rotation(video)
        if media.rotation in (90, 270):
            media.width, media.height = media.height, media.width
        if media.duration <= 0:
            media.duration = _num(video.get("duration"))
        # Одна картинка без движения — считаем фото.
        if media.duration <= 0.1 and video.get("nb_frames") in (None, "1", 1):
            media.kind = "image"
            media.fps = 0.0
        if media.kind == "image" and media.duration <= 0.1:
            media.duration = 5.0

    if audio:
        media.acodec = audio.get("codec_name") or ""
        media.has_audio = True
        if media.kind == "video" and not video:
            media.kind = "audio"
    if video is None and audio is None:
        kind_by_ext = Path(path).suffix.lower()
        media.kind = "audio" if kind_by_ext in AUDIO_EXT else "video"
    return media


def collect_sources(tools: Tools, folder: str, recursive: bool = True) -> tuple[list[Media], list[Media]]:
    """Раскладывает содержимое папки на клипы и треки."""
    root = Path(folder)
    files = sorted(p for p in (root.rglob("*") if recursive else root.glob("*")) if p.is_file())
    clips: list[Media] = []
    tracks: list[Media] = []
    for f in files:
        ext = f.suffix.lower()
        if ext not in VIDEO_EXT | AUDIO_EXT | IMAGE_EXT:
            continue
        try:
            m = probe(tools, str(f))
        except Exception:
            continue
        if m.kind == "audio":
            tracks.append(m)
        elif m.duration > 0.05:
            clips.append(m)
    return clips, tracks
