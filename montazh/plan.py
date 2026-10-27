"""Планировщик монтажа: раскладывает шоты по битам музыки.

На выходе — EDL (edit decision list) в JSON: что, откуда, с какой скоростью и
зумом, сколько кадров занимает. Кадры считаем целыми: склейка всегда попадает
на кадровую сетку, поэтому видео и музыка не расходятся.

Три стиля:
  dynamic   — склейки на каждый бит/долю, резкие каты, короткие планы;
  cinematic — длинные планы (такт-два), наплывы, лёгкий зум;
  slideshow — фото/статика под музыку, Ken Burns, длинные плавные переходы.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio import TrackAnalysis
from .shots import Shot

# Ниже этого замедления план не растягиваем: движение становится «желейным».
MIN_SPEED = 0.75

STYLES = {
    "dynamic": {
        "slots_hot": [1, 1, 2],
        "slots_calm": [2, 4],
        "hot_threshold": 0.55,
        "speed_jitter": 0.06,
        "zoom_chance": 0.25,
        "zoom_amount": 0.05,
        "transition": "cut",
        "transition_dur": 0.0,
        "prefer_motion": True,
    },
    "cinematic": {
        "slots_hot": [4, 4, 8],
        "slots_calm": [8, 8],
        "hot_threshold": 0.5,
        "speed_jitter": 0.02,
        "zoom_chance": 0.75,
        "zoom_amount": 0.035,
        "transition": "fade",
        "transition_dur": 0.30,
        "prefer_motion": False,
    },
    "slideshow": {
        "slots_hot": [4, 8],
        "slots_calm": [8, 8, 8],
        "hot_threshold": 0.5,
        "speed_jitter": 0.0,
        "zoom_chance": 1.0,
        "zoom_amount": 0.10,
        "transition": "fade",
        "transition_dur": 0.55,
        "prefer_motion": False,
    },
}

ASPECTS = {
    "1:1": (1, 1),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "4:5": (4, 5),
    "source": None,
}


@dataclass
class Segment:
    i: int
    src: str
    shot: int
    src_in: float
    src_dur: float
    speed: float
    frames: int
    zoom_from: float = 1.0
    zoom_to: float = 1.0
    hflip: bool = False
    transition: str = "cut"
    trans_dur: float = 0.0
    strength: float = 0.0
    # Хвост под наплыв: он съедается переходом, поэтому в длительность не входит.
    tail_frames: int = 0

    def out_dur(self, fps: int) -> float:
        return self.frames / fps

    @property
    def render_frames(self) -> int:
        return self.frames + self.tail_frames


@dataclass
class Plan:
    style: str
    fps: int
    width: int
    height: int
    duration: float
    music_path: str
    music_start: float
    music_duration: float
    bpm: float
    segments: list[Segment] = field(default_factory=list)
    cuts: int = 0
    unique_shots: int = 0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), "utf-8")

    @property
    def total_frames(self) -> int:
        return sum(s.frames for s in self.segments)

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        data = json.loads(Path(path).read_text("utf-8"))
        segs = [Segment(**s) for s in data.pop("segments", [])]
        return cls(segments=segs, **data)


def _resolution(sizes: list[tuple[int, int]], aspect: str, height: int | None) -> tuple[int, int]:
    ratio = ASPECTS.get(aspect)
    if ratio is None:
        from collections import Counter

        base = Counter(sizes).most_common(1)[0][0] if sizes else (1920, 1080)
        w, h = base
        return int(w // 2 * 2), int(h // 2 * 2)

    rw, rh = ratio
    if height:
        h = height
        w = int(round(h * rw / rh))
    else:
        long_side = 1080
        if rw >= rh:
            w, h = long_side, int(round(long_side * rh / rw))
        else:
            h, w = long_side * 16 // 9, long_side
    return int(w // 2 * 2), int(h // 2 * 2)


def _shot_weight(shot: Shot, style: dict, uses: int, recent: dict[int, int], now: int,
                 allow_dark: bool) -> float:
    if shot.duration < 0.32:
        return 0.0
    if shot.is_dark and not allow_dark:
        return 0.0
    base = max(shot.score, 0.05) ** 1.6
    if style["prefer_motion"]:
        base *= 1.0 + 1.5 * min(shot.motion * 12.0, 1.0)
    else:
        base *= 1.0 + 0.8 * min(shot.sharpness, 1.5) + 0.5 * min(shot.contrast, 1.5)
    base /= 1.0 + 2.5 * uses
    if shot.index in recent:
        base *= 0.05 if now - recent[shot.index] < 4 else 0.35
    return base


def _pick_shot(shots: list[Shot], style: dict, required: float, rng: random.Random,
               uses: dict[int, int], recent: dict[int, int], now: int,
               allow_dark: bool) -> Shot | None:
    for attempt in range(60):
        cands = []
        weights = []
        lenient = attempt > 20
        for shot in shots:
            avail = shot.duration - 0.24
            if avail <= 0.15:
                continue
            if avail < required * 0.72 and not lenient:
                continue
            w = _shot_weight(shot, style, uses.get(shot.index, 0), recent, now, allow_dark)
            if w <= 0:
                continue
            cands.append(shot)
            weights.append(w)
        if not cands:
            if allow_dark:
                return None
            allow_dark = True
            continue
        total = sum(weights)
        x = rng.random() * total
        acc = 0.0
        chosen = cands[-1]
        for shot, w in zip(cands, weights):
            acc += w
            if x <= acc:
                chosen = shot
                break
        return chosen
    return None


def _choose_window(track: TrackAnalysis, target: float | None, music_start: float | None) -> tuple[float, float]:
    beats = track.beats
    if not beats:
        return 0.0, min(target or 60.0, track.duration)
    if music_start is None:
        # Стартуем с дропа или с первого «не тихого» такта — без длинного интро.
        cand = track.drop
        if not cand:
            for sec in track.sections:
                if sec["level"] != "quiet" and sec["end"] - sec["start"] > 10:
                    cand = sec["start"]
                    break
        music_start = max(cand, 0.0)
        downbeats = [b for b in track.downbeats if b >= music_start - track.beat_period * 2]
        if downbeats:
            # Ближайшая доля, а не следующая: иначе теряем начало фразы.
            music_start = min(downbeats, key=lambda b: abs(b - music_start))
    if target is None:
        target = min(track.duration - music_start, 150.0)
    target = max(1.5, min(target, track.duration - music_start))
    return music_start, target


def build_plan(shots: list[Shot], track: TrackAnalysis, *, style: str = "dynamic",
               target: float | None = None, fps: int = 30, aspect: str = "source",
               height: int | None = None, music_start: float | None = None,
               seed: int = 0, max_reuse: int = 4,
               sizes: list[tuple[int, int]] | None = None) -> Plan:
    if style not in STYLES:
        raise ValueError(f"Неизвестный стиль {style!r}. Доступны: {', '.join(STYLES)}")
    presets = STYLES[style]
    rng = random.Random(seed)

    start, target = _choose_window(track, target, music_start)
    width, height_px = _resolution(sizes or [], aspect, height)

    beats = [b for b in track.beats if b >= start - 1e-6]
    energy = track.beat_energy
    beat_index = {round(b, 3): i for i, b in enumerate(track.beats)}

    if len(beats) < 4:
        raise ValueError("В выбранном фрагменте музыки слишком мало битов")

    # 1. Слоты: сколько битов держим каждый план.
    beat_len = track.beat_period or (float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5)
    usable = [s.duration - 0.24 for s in shots if s.duration > 0.4]
    max_slot = (max(usable) / MIN_SPEED) if usable else 1.0

    slots: list[int] = []
    cursor = 0
    rng_local = random.Random(seed + 1)
    trimmed_slots = False
    while True:
        if cursor >= len(beats):
            break
        idx_in_track = beat_index.get(round(beats[cursor], 3), cursor)
        e = energy[idx_in_track] if idx_in_track < len(energy) else 0.5
        if e >= presets["hot_threshold"]:
            choices = presets["slots_hot"]
        else:
            choices = presets["slots_calm"]
        # Слот не длиннее, чем может закрыть самый длинный шот.
        allowed = [n for n in choices if n * beat_len <= max_slot]
        if not allowed:
            allowed = [min(choices)]
            trimmed_slots = True
        elif len(allowed) < len(choices):
            trimmed_slots = True
        length = rng_local.choice(allowed)
        slots.append(min(length, len(beats) - cursor))
        cursor += slots[-1]
        if beats[cursor - 1] - start >= target:
            break

    # 2. Границы слотов в кадрах: всегда на кадровой сетке.
    boundaries = [0]
    acc = 0
    for n in slots:
        acc += n
        if acc >= len(beats):
            beat_time = start + target
        else:
            beat_time = beats[acc]
        frame = int(round((beat_time - start) * fps))
        frame = max(frame, boundaries[-1] + 2)
        boundaries.append(frame)
    # Отрезаем всё, что вылезло за целевую длительность.
    limit = int(round(target * fps))
    boundaries = [b for b in boundaries if b <= limit] or [0, limit]
    if boundaries[-1] < limit:
        boundaries.append(limit)

    frames_list = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    frames_list = [f for f in frames_list if f >= 2]
    duration = sum(frames_list) / fps

    # 3. Подбор шотов под слоты.
    uses: dict[int, int] = {}
    recent: dict[int, int] = {}
    segments: list[Segment] = []
    hot = presets["slots_hot"][0]
    for i, frames in enumerate(frames_list):
        out_dur = frames / fps
        t_in_track = start + sum(frames_list[:i]) / fps
        idx_in_track = min(range(len(beats)), key=lambda k: abs(beats[k] - t_in_track)) if beats else 0
        e = energy[idx_in_track] if idx_in_track < len(energy) else 0.5

        transition = presets["transition"] if i < len(frames_list) - 1 else "cut"
        trans_dur = presets["transition_dur"] if transition != "cut" else 0.0
        # Переход не должен съесть больше четверти короткого плана.
        trans_dur = min(trans_dur, out_dur * 0.25) if trans_dur else 0.0
        tail = int(round(trans_dur * fps)) if trans_dur else 0

        speed = 1.0 + rng.uniform(-presets["speed_jitter"], presets["speed_jitter"])
        need_dur = (frames + tail) / fps
        required = need_dur * speed
        shot = _pick_shot(shots, presets, required, rng, uses, recent, i, allow_dark=False)
        if shot is None:
            raise ValueError("Не хватает материала под выбранную длительность — добавьте клипы или сократите ролик")

        avail = shot.duration - 0.18
        if avail < required:
            speed = max(MIN_SPEED, avail / need_dur)
            required = need_dur * speed
        src_dur = min(required, avail)
        latest = shot.start + shot.duration - 0.12 - src_dur
        earliest = shot.start + 0.12
        src_in = rng.uniform(earliest, max(earliest, latest)) if latest > earliest else earliest

        zoom_to = 1.0
        zoom_from = 1.0
        if style == "slideshow":
            direction = 1 if i % 2 == 0 else -1
            amount = presets["zoom_amount"]
            if direction > 0:
                zoom_from, zoom_to = 1.0, 1.0 + amount
            else:
                zoom_from, zoom_to = 1.0 + amount, 1.0
        elif rng.random() < presets["zoom_chance"]:
            amount = presets["zoom_amount"] * (1.4 if e > 0.7 else 1.0)
            if rng.random() < 0.5:
                zoom_from, zoom_to = 1.0, 1.0 + amount
            else:
                zoom_from, zoom_to = 1.0 + amount, 1.0

        transition = presets["transition"] if i < len(frames_list) - 1 else "cut"
        trans_dur = presets["transition_dur"] if transition != "cut" else 0.0
        # Переход не должен съесть больше четверти короткого плана.
        trans_dur = min(trans_dur, out_dur * 0.25) if trans_dur else 0.0

        segments.append(
            Segment(
                i=i,
                src=shot.src,
                shot=shot.index,
                src_in=round(src_in, 3),
                src_dur=round(src_dur + 0.2, 3),
                speed=round(speed, 4),
                frames=frames,
                zoom_from=round(zoom_from, 4),
                zoom_to=round(zoom_to, 4),
                hflip=uses.get(shot.index, 0) >= 2 and rng.random() < 0.6,
                transition=transition,
                trans_dur=round(trans_dur, 3),
                strength=round(float(e), 3),
                tail_frames=tail,
            )
        )
        uses[shot.index] = uses.get(shot.index, 0) + 1
        recent[shot.index] = i

    plan = Plan(
        style=style,
        fps=fps,
        width=width,
        height=height_px,
        duration=round(duration, 3),
        music_path=track.path,
        music_start=round(start, 3),
        music_duration=round(duration, 3),
        bpm=track.bpm,
        segments=segments,
        cuts=len(segments),
        unique_shots=len(uses),
    )
    _notes(plan, shots, uses, max_reuse)
    if trimmed_slots:
        plan.notes.append(
            "Часть планов короче задуманного: шоты не тянут длинный слот (материал короткий)"
        )
    return plan


def _notes(plan: Plan, shots: list[Shot], uses: dict[int, int], max_reuse: int) -> None:
    if not plan.segments:
        return
    per_sec = plan.cuts / max(plan.duration, 0.1)
    plan.notes.append(f"Средняя длина плана — {plan.duration / plan.cuts:.2f} c ({per_sec:.1f} склейки в секунду)")
    heavy = [idx for idx, n in uses.items() if n >= 3]
    if heavy:
        plan.notes.append(f"{len(heavy)} шотов использованы 3+ раза — в кадре появится повтор")
    if plan.unique_shots < plan.cuts * 0.5:
        plan.notes.append("Уникальных шотов меньше половины склеек: материал короткий для такой длительности")


def summarize_plan(plan: Plan) -> dict:
    segs = plan.segments
    return {
        "style": plan.style,
        "duration": plan.duration,
        "cuts": plan.cuts,
        "cuts_per_sec": round(plan.cuts / plan.duration, 2) if plan.duration else 0,
        "avg_shot": round(plan.duration / plan.cuts, 2) if plan.cuts else 0,
        "unique_shots": plan.unique_shots,
        "fps": plan.fps,
        "size": f"{plan.width}x{plan.height}",
        "music_start": plan.music_start,
        "bpm": plan.bpm,
        "notes": plan.notes,
    }
