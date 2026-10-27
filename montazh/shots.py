"""Разбор исходного видео на шоты (планы) по склейкам.

Считаем покадровую разницу на уменьшённых серых кадрах: так детект не зависит
от кодека и работает на любом материале. Заодно собираем статистику плана —
яркость, контраст, движение, резкость — по ней планировщик выбирает, что
ставить в мощные и в спокойные места.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .env import run_bytes


@dataclass
class Shot:
    src: str
    start: float
    end: float
    brightness: float = 0.0   # 0..1
    contrast: float = 0.0     # std яркости / 64
    motion: float = 0.0       # средняя разница кадров, 0..1
    sharpness: float = 0.0    # дисперсия лапласиана, нормированная
    index: int = 0
    score: float = 0.0        # «годность» для динамичного монтажа
    dark: bool = False        # темнее остального материала этого источника

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def is_dark(self) -> bool:
        return self.dark or self.brightness < 0.06

    @property
    def is_blurry(self) -> bool:
        return self.sharpness < 0.04

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("start", "end", "brightness", "contrast", "motion", "sharpness", "score"):
            d[key] = round(float(d[key]), 4)
        return d


def decode_gray(src: str, ffmpeg: str, fps: float = 10.0, size: int = 192) -> tuple[np.ndarray, float]:
    """Кадры в сером, уменьшенные: (N, size, size) uint8 и фактический fps."""
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(src),
        "-vf", f"fps={fps},scale={size}:{size}:flags=area,format=gray",
        "-f", "rawvideo", "-",
    ]
    data = run_bytes(cmd)
    frame_size = size * size
    count = len(data) // frame_size
    if count == 0:
        return np.zeros((0, size, size), dtype=np.uint8), fps
    arr = np.frombuffer(data[: count * frame_size], dtype=np.uint8)
    return arr.reshape(count, size, size), fps


def _frame_diffs(frames: np.ndarray) -> np.ndarray:
    a = frames[1:].astype(np.int16)
    b = frames[:-1].astype(np.int16)
    return np.abs(a - b).mean(axis=(1, 2)) / 255.0


def _sharpness(frame: np.ndarray) -> float:
    f = frame.astype(np.float32) / 255.0
    lap = (
        -4 * f[1:-1, 1:-1]
        + f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:]
    )
    return float(lap.var()) / 0.02


def detect_cuts(diffs: np.ndarray, fps: float, threshold: float | None = None,
                min_shot: float = 0.35) -> list[int]:
    """Индексы кадров, на которых начинается новый план."""
    if len(diffs) < 3:
        return []
    if threshold is None:
        median = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - median))) or 0.01
        threshold = max(0.055, median + 5.0 * mad)

    gap = max(1, int(round(min_shot * fps)))
    cuts: list[int] = []
    for i in range(1, len(diffs) - 1):
        if diffs[i] < threshold:
            continue
        if diffs[i] < diffs[i - 1] or diffs[i] < diffs[i + 1]:
            continue
        if cuts and i - cuts[-1] < gap:
            # соседний пик — оставляем более сильный
            if diffs[i] > diffs[cuts[-1] - 1]:
                cuts[-1] = i + 1
            continue
        cuts.append(i + 1)
    return cuts


def analyze_shots(src: str, ffmpeg: str, fps: float = 30.0, size: int = 160,
                  threshold: float | None = None, min_shot: float = 0.35,
                  max_shots: int = 400, still_duration: float = 6.0) -> list[Shot]:
    frames, real_fps = decode_gray(src, ffmpeg, fps=fps, size=size)
    if len(frames) < 3:
        # Фото: один «план» на всю длину, его растянет Ken Burns.
        if len(frames) == 1:
            frame = frames[0]
            shot = Shot(
                src=str(src), start=0.0, end=still_duration,
                brightness=float(frame.mean()) / 255.0,
                contrast=float(frame.std()) / 64.0,
                motion=0.0,
                sharpness=min(_sharpness(frame), 5.0),
                index=0,
            )
            shot.score = max(_score(shot), 0.4)
            return [shot]
        return []
    diffs = _frame_diffs(frames)
    cuts = detect_cuts(diffs, real_fps, threshold=threshold, min_shot=min_shot)
    bounds = [0, *cuts, len(frames)]
    bounds = sorted(set(bounds))

    shots: list[Shot] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 2:
            continue
        chunk = frames[a:b]
        mean = float(chunk.mean()) / 255.0
        contrast = float(chunk.std()) / 64.0
        inner = _frame_diffs(chunk) if len(chunk) > 1 else np.zeros(1)
        motion = float(inner.mean())
        sharp = float(np.mean([_sharpness(f) for f in chunk[:: max(1, len(chunk) // 5)]][:6]))
        shots.append(
            Shot(
                src=str(src),
                start=round(a / real_fps, 4),
                end=round(b / real_fps, 4),
                brightness=mean,
                contrast=contrast,
                motion=motion,
                sharpness=min(sharp, 5.0),
                index=len(shots),
            )
        )

    # Темнота — понятие относительное: Mr. Robot темнее, чем отпуск на море.
    if shots:
        source_median = float(np.median([s.brightness for s in shots]))
        limit = max(0.05, source_median * 0.45)
        for shot in shots:
            shot.dark = shot.brightness < limit

    for shot in shots:
        shot.score = _score(shot)

    # Слишком много шотов (нарезка каждые 0.2 c) — оставляем самые длинные.
    if len(shots) > max_shots:
        keep = sorted(sorted(shots, key=lambda s: s.duration, reverse=True)[:max_shots], key=lambda s: s.start)
        for i, shot in enumerate(keep):
            shot.index = i
        shots = keep
    return shots


def _score(shot: Shot) -> float:
    """Пригодность плана для динамичного монтажа: живой, резкий, не чёрный."""
    if shot.is_dark or shot.duration < 0.3:
        return 0.0
    return float(
        0.35 * min(shot.motion * 6.0, 1.0)
        + 0.35 * min(shot.sharpness * 2.0, 1.0)
        + 0.20 * min(shot.contrast * 1.5, 1.0)
        + 0.10 * min(shot.duration / 3.0, 1.0)
    )


def shots_from_entries(entries: list[dict]) -> list[Shot]:
    """Восстанавливает шоты из JSON (кеш на диске)."""
    out = []
    for e in entries:
        s = Shot(
            src=e["src"], start=float(e["start"]), end=float(e["end"]),
            brightness=float(e.get("brightness", 0)), contrast=float(e.get("contrast", 0)),
            motion=float(e.get("motion", 0)), sharpness=float(e.get("sharpness", 0)),
            index=int(e.get("index", len(out))), score=float(e.get("score", 0)),
        )
        out.append(s)
    return out


def summarize(shots: list[Shot]) -> dict:
    if not shots:
        return {"count": 0, "total": 0.0}
    durations = np.array([s.duration for s in shots])
    return {
        "count": len(shots),
        "total": round(float(durations.sum()), 2),
        "median": round(float(np.median(durations)), 3),
        "min": round(float(durations.min()), 3),
        "max": round(float(durations.max()), 3),
        "dark": int(sum(s.is_dark for s in shots)),
        "sources": len({s.src for s in shots}),
        "path": str(Path(shots[0].src).parent),
    }
