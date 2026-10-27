"""Анализ музыки: биты, темп, доли, секции.

Никаких тяжёлых зависимостей — только numpy. Схема:
  1) декодируем трек в моно 22050 Гц через ffmpeg;
  2) считаем спектральный поток (onset envelope) и низкочастотный поток (бочка);
  3) темп и фазу ищем гребёнкой по сетке битов: автокорреляция даёт стартовую
     оценку, динамическое программирование — грубый путь по долям, линейная
     подгонка — точный период, финальная гребёнка — точную фазу;
  4) сетка битов идеально ровная (важно для склеек), но если трек «живой» и
     сетка сильно хуже DP-пути — отдаём DP-путь;
  5) сильно/слабо звучащие биты и скачки энергии дают секции (куплет/припев/дроп).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .env import Tools, run_bytes

SR = 22050
HOP = 256
N_FFT = 1024


# ---------------------------------------------------------------- декодирование


def load_mono(path: str, ffmpeg: str, sr: int = SR, start: float | None = None,
              duration: float | None = None) -> np.ndarray:
    """Декодирует файл в моно float32 через pipe ffmpeg."""
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start:.4f}"]
    if duration:
        cmd += ["-t", f"{duration:.4f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    data = run_bytes(cmd)
    return np.frombuffer(data, dtype=np.float32).copy()


def _mag_chunk(x: np.ndarray, start: int, count: int, n_fft: int, hop: int, win: np.ndarray) -> np.ndarray:
    end = (start + count - 1) * hop + n_fft
    seg = x[start * hop:end]
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    frames = np.lib.stride_tricks.sliding_window_view(seg, n_fft)[::hop][:count]
    return np.abs(np.fft.rfft(frames * win, axis=1)).astype(np.float32)


def onset_envelopes(x: np.ndarray, sr: int = SR, n_fft: int = N_FFT, hop: int = HOP):
    """Возвращает (полный поток, низкочастотный поток, кадры в секунду)."""
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    if n_frames < 8:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), sr / hop

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    low_bins = freqs <= 160.0
    full = np.zeros(n_frames, dtype=np.float32)
    low = np.zeros(n_frames, dtype=np.float32)

    chunk = 4096
    prev_mag = None
    for start in range(0, n_frames, chunk):
        count = min(chunk, n_frames - start)
        mag = _mag_chunk(x, start, count, n_fft, hop, win)
        logmag = np.log1p(mag * 12.0)
        prev = prev_mag if prev_mag is not None else np.zeros_like(logmag[:1])
        diff = np.maximum(np.diff(np.vstack([prev, logmag]), axis=0), 0.0)
        body = diff[:count]
        full[start:start + count] = body.sum(axis=1)
        low[start:start + count] = body[:, low_bins].sum(axis=1)
        prev_mag = logmag[-1:]

    fps = sr / hop
    for arr in (full, low):
        scale = np.percentile(arr, 99) + 1e-9
        arr /= scale
    return full, low, fps


def env_at(env: np.ndarray, times: np.ndarray, fps: float, radius: float = 0.03) -> np.ndarray:
    """Значение потока в момент времени (максимум в окне ±radius)."""
    times = np.asarray(times, dtype=np.float64)
    if not len(times) or not len(env):
        return np.zeros(0, dtype=np.float32)
    reach = max(1, int(round(radius * fps)))
    idx = np.round(times * fps).astype(np.int64)
    idx = np.clip(idx, reach, max(reach, len(env) - reach - 1))
    stack = np.stack([env[np.clip(idx + off, 0, len(env) - 1)] for off in range(-reach, reach + 1)])
    return stack.max(axis=0)


# ---------------------------------------------------------------- темп и биты


def _autocorr_period(env: np.ndarray, fps: float, bpm_min: float, bpm_max: float) -> float:
    e = env - env.mean()
    n = len(e)
    if n < 64 or not np.any(e):
        return 0.0
    size = 1 << int(math.ceil(math.log2(2 * n)))
    spec = np.fft.rfft(e, size)
    ac = np.fft.irfft(spec * np.conj(spec))[:n].astype(np.float64)
    ac /= np.maximum(np.arange(n, 0, -1), 1)

    lag_min = max(2, int(round(60.0 * fps / bpm_max)))
    lag_max = min(n - 2, int(round(60.0 * fps / bpm_min)))
    if lag_max <= lag_min:
        return 0.0
    window = ac[lag_min:lag_max]
    best = int(np.argmax(window))
    lag = float(lag_min + best)
    if 0 < best < len(window) - 1:
        y0, y1, y2 = window[best - 1], window[best], window[best + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            lag += 0.5 * (y0 - y2) / denom
    return lag


def track_beats(env: np.ndarray, period: float, tightness: float = 90.0) -> list[int]:
    """Динамическое программирование: путь по самым сильным долям с ровным шагом."""
    n = len(env)
    if n < 8 or period < 2:
        return []
    score = env / (env.std() + 1e-9)
    lo_off = max(1, int(round(period * 0.55)))
    hi_off = max(lo_off + 1, int(round(period * 1.9)))
    hi_off = min(hi_off, n - 1)
    deltas = np.arange(1, hi_off + 1)
    txcost = -tightness * np.log(deltas / period) ** 2
    txcost = np.where(deltas < lo_off, -1e9, txcost)

    cum = np.zeros(n)
    back = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        lo = i - hi_off
        if lo < 0:
            cum[i] = score[i]
            continue
        window = cum[lo:i]
        costs = txcost[: len(window)][::-1]
        cand = window + costs
        k = int(np.argmax(cand))
        cum[i] = score[i] + cand[k]
        back[i] = lo + k

    tail_start = max(0, n - int(period * 1.2))
    end = tail_start + int(np.argmax(cum[tail_start:]))
    path = []
    while end >= 0:
        path.append(int(end))
        end = int(back[end])
    path.reverse()
    return path


def _interp(env: np.ndarray, times: np.ndarray, fps: float) -> np.ndarray:
    """Линейная интерполяция потока в произвольные моменты времени."""
    if not len(times) or not len(env):
        return np.zeros(0, dtype=np.float32)
    idx = np.clip(np.asarray(times, dtype=np.float64) * fps, 0, len(env) - 1.0001)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(env) - 1)
    w = idx - i0
    return (env[i0] * (1 - w) + env[i1] * w).astype(np.float32)


def _comb(env: np.ndarray, fps: float, bpm: float, phase: float, duration: float) -> float:
    """Контраст сетки: на битах поток должен быть сильнее, чем между битами."""
    period = 60.0 / bpm
    if period <= 0 or phase >= duration:
        return -1.0
    pos = phase + np.arange(int((duration - phase) / period) + 1) * period
    pos = pos[(pos >= 0) & (pos < duration)]
    if len(pos) < 4:
        return -1.0
    on = _interp(env, pos, fps).mean()
    off = _interp(env, pos + period / 2.0, fps).mean()
    return float(on - 0.5 * off)


def estimate_tempo(env: np.ndarray, fps: float, bpm_min: float = 68.0,
                   bpm_max: float = 190.0) -> dict:
    """Ищет (bpm, период в секундах, фаза в секундах, качество сетки)."""
    duration = len(env) / fps
    lag = _autocorr_period(env, fps, bpm_min, bpm_max)
    if not lag:
        return {"bpm": 0.0, "period": 0.0, "phase": 0.0, "score": 0.0}

    # Грубая сетка из автокорреляции, уточнение пути DP и линейная подгонка.
    path = track_beats(env, lag)
    if len(path) >= 8:
        times = np.array(path, dtype=np.float64) / fps
        k = np.round((times - times[0]) / (lag / fps))
        slope, intercept = np.polyfit(k, times, 1)
        if slope > 0.15:
            bpm0 = 60.0 / slope
            phase0 = float(intercept % slope)
        else:
            bpm0, phase0 = 60.0 * fps / lag, 0.0
    else:
        bpm0, phase0 = 60.0 * fps / lag, 0.0

    # Гребёнка: сначала грубо, потом точно вокруг найденного.
    best = (_comb(env, fps, bpm0, phase0, duration), bpm0, phase0)
    step = max(0.05, bpm0 * 0.004)
    for bpm in np.arange(bpm0 - 6 * step, bpm0 + 6 * step + 1e-9, step):
        period = 60.0 / bpm
        for phase in np.arange(phase0 - 0.10, phase0 + 0.10 + 1e-9, 0.005):
            phase = phase % period
            score = _comb(env, fps, bpm, phase, duration)
            if score > best[0]:
                best = (score, float(bpm), float(phase))
    for bpm in np.arange(best[1] - step, best[1] + step + 1e-9, step / 10):
        period = 60.0 / bpm
        for phase in np.arange(best[2] - 0.02, best[2] + 0.02 + 1e-9, 0.001):
            phase = phase % period
            score = _comb(env, fps, bpm, phase, duration)
            if score > best[0]:
                best = (score, float(bpm), float(phase))

    score, bpm, phase = best
    return {"bpm": bpm, "period": 60.0 / bpm if bpm else 0.0, "phase": phase, "score": score}


def build_grid(period: float, phase: float, duration: float) -> np.ndarray:
    if period <= 0:
        return np.array([])
    count = int((duration - phase) / period) + 1
    beats = phase + np.arange(max(count, 0)) * period
    return beats[(beats >= 0) & (beats < duration)]


# ---------------------------------------------------------------- секции


def energy_curve(x: np.ndarray, sr: int = SR, window: float = 0.5) -> tuple[np.ndarray, float]:
    """RMS по окнам — для поиска куплетов/припевов и дропа."""
    size = max(1, int(window * sr))
    count = max(1, len(x) // size)
    trimmed = x[: count * size].astype(np.float32)
    frames = trimmed.reshape(count, size)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    return rms, window


def find_sections(rms: np.ndarray, window: float, min_len: float = 8.0) -> list[dict]:
    """Куски с разной громкостью: тихо/средне/мощно."""
    if len(rms) < 4:
        return [{"start": 0.0, "end": len(rms) * window, "level": "mid", "energy": 1.0}]
    smooth = np.convolve(rms, np.ones(5) / 5, mode="same")
    median = float(np.median(smooth[smooth > 0])) if np.any(smooth > 0) else 1.0
    median = median or 1.0
    levels = smooth / median
    labels = np.where(levels < 0.65, "quiet", np.where(levels > 1.25, "loud", "mid"))

    sections: list[dict] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            sections.append(
                {
                    "start": round(start * window, 3),
                    "end": round(i * window, 3),
                    "level": str(labels[start]),
                    "energy": round(float(levels[start:i].mean()), 3),
                }
            )
            start = i

    merged: list[dict] = []
    for sec in sections:
        if merged and sec["end"] - sec["start"] < min_len:
            merged[-1]["end"] = sec["end"]
        elif merged and merged[-1]["end"] - merged[-1]["start"] < min_len:
            sec["start"] = merged[-1]["start"]
            merged[-1] = sec
        else:
            merged.append(sec)
    return merged


def first_drop(rms: np.ndarray, window: float) -> float:
    """Первый момент, где трек выходит на свою «мощную» громкость."""
    if len(rms) < 24:
        return 0.0
    smooth = np.convolve(rms, np.ones(8) / 8, mode="same")
    live = smooth[smooth > 0]
    if not len(live):
        return 0.0
    threshold = float(np.median(live)) * 1.25
    loud = smooth > threshold
    need = 16
    for i in range(len(loud) - need):
        if loud[i:i + need].mean() > 0.8:
            t = i * window
            # Не считаем дропом самое начало трека.
            return round(t, 3) if t > 2.0 * window else 0.0
    return 0.0


# ---------------------------------------------------------------- итог анализа


@dataclass
class TrackAnalysis:
    path: str
    duration: float
    bpm: float
    beat_period: float
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    beat_energy: list[float] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    drop: float = 0.0
    grid: str = "grid"  # grid | dp — какой путь битов взят
    sr: int = SR
    hop: int = HOP

    def to_json(self) -> dict:
        d = asdict(self)
        d["beats"] = [round(b, 3) for b in self.beats]
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), "utf-8")

    def section_at(self, t: float) -> dict:
        for sec in self.sections:
            if sec["start"] <= t < sec["end"]:
                return sec
        return self.sections[-1] if self.sections else {"level": "mid", "energy": 1.0}


def analyze_track(tools: Tools, path: str, start: float | None = None,
                  duration: float | None = None) -> TrackAnalysis:
    x = load_mono(path, tools.ffmpeg, start=start, duration=duration)
    total = len(x) / SR
    full, low, fps = onset_envelopes(x)
    info = estimate_tempo(full, fps)
    bpm, period, phase = info["bpm"], info["period"], info["phase"]

    grid = build_grid(period, phase, total)
    mode = "grid"
    if not len(grid):
        mode = "dp"
        grid = np.array([])

    # Живой трек: если ровная сетка заметно хуже пути DP — берём DP.
    if mode == "grid" and period:
        dp_path = track_beats(full, period * fps)
        if len(dp_path) > 8:
            dp_times = np.array(dp_path) / fps
            grid_score = float(env_at(full, grid, fps, radius=0.06).mean())
            dp_score = float(env_at(full, dp_times, fps, radius=0.06).mean())
            if grid_score < dp_score * 0.72:
                grid, mode = dp_times, "dp"

    beats = [round(float(b), 4) for b in grid]
    strengths: list[float] = []
    if len(grid):
        raw = env_at(full, np.array(grid), fps, radius=0.03)
        top = float(raw.max()) or 1.0
        strengths = [round(float(v) / top, 3) for v in raw]

    # Доли: из четырёх фаз берём ту, где сильнее всего бочка.
    downbeats: list[float] = []
    if len(grid):
        low_at = env_at(low, np.array(grid), fps, radius=0.04)
        best_phase, best_val = 0, -1.0
        for ph in range(4):
            part = low_at[ph::4]
            val = float(part.mean()) if len(part) else -1.0
            if val > best_val:
                best_phase, best_val = ph, val
        downbeats = [round(float(beats[i]), 4) for i in range(best_phase, len(beats), 4)]

    rms, win = energy_curve(x)
    sections = find_sections(rms, win)
    drop = first_drop(rms, win)

    return TrackAnalysis(
        path=str(path),
        duration=round(total, 3),
        bpm=round(bpm, 2),
        beat_period=round(1.0 / (bpm / 60.0), 4) if bpm else 0.0,
        beats=beats,
        downbeats=downbeats,
        beat_energy=strengths,
        sections=sections,
        drop=drop,
        grid=mode,
    )
