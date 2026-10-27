"""Тесты скилла на синтетическом материале.

Материал генерируется ffmpeg'ом: шесть цветных планов по 1.2 c с движущимся
квадратом и клик-трек на 120 BPM. Так тест не зависит от файлов пользователя
и проверяет весь путь: анализ музыки → детект склеек → план → рендер.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from montazh.audio import analyze_track  # noqa: E402
from montazh.env import find_tools  # noqa: E402
from montazh.plan import build_plan  # noqa: E402
from montazh.render import render  # noqa: E402
from montazh.shots import analyze_shots  # noqa: E402

BPM = 120.0
SHOT_LEN = 1.2
COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]


def make_click_track(path: Path, seconds: float = 9.0, sr: int = 44100) -> None:
    """Кик на каждой доле 120 BPM + тихий хэт между долями."""
    n = int(seconds * sr)
    x = np.zeros(n, dtype=np.float32)
    period = 60.0 / BPM
    kick_len = int(0.12 * sr)
    t = np.arange(kick_len) / sr
    kick = (np.sin(2 * np.pi * 70 * t) * np.exp(-t * 35)).astype(np.float32)

    k = 0
    while k * period < seconds - 0.2:
        start = int(k * period * sr)
        end = min(start + kick_len, n)
        x[start:end] += kick[: end - start]
        k += 1

    # хэт на слабой доле — чтобы трек не был монотонным кликом
    hat_len = int(0.03 * sr)
    rng = np.random.default_rng(0)
    hat = (rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / (0.006 * sr))).astype(np.float32) * 0.15
    k = 0
    while (k + 0.5) * period < seconds - 0.2:
        start = int((k + 0.5) * period * sr)
        end = min(start + hat_len, n)
        x[start:end] += hat[: end - start]
        k += 1

    # лёгкий бас-аккорд на такт
    bar = period * 4
    k = 0
    while k * bar < seconds - 0.4:
        start = int(k * bar * sr)
        dur = int(bar * 0.9 * sr)
        end = min(start + dur, n)
        tt = np.arange(end - start) / sr
        tone = (np.sin(2 * np.pi * 110 * tt) * 0.12).astype(np.float32)
        x[start:end] += tone
        k += 1

    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def make_clips(path: Path, tools) -> int:
    """Шесть цветных планов, склеенных подряд: на них проверяем детект склеек."""
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, color in enumerate(COLORS):
            part = Path(tmp) / f"part{i}.mp4"
            cmd = [
                tools.ffmpeg, "-y", "-v", "error",
                "-f", "lavfi", "-i",
                f"color=c={color}:s=320x320:r=30:d={SHOT_LEN}",
                "-vf",
                "drawbox=x='mod(t*180,260)':y='mod(t*120,260)':w=60:h=60:color=white@0.9:t=fill",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-an", str(part),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            parts.append(part)
        listing = Path(tmp) / "list.txt"
        listing.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts), "utf-8")
        subprocess.run(
            [tools.ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", str(path)],
            check=True, capture_output=True,
        )
    return len(COLORS)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tools = find_tools()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.video = cls.dir / "clips.mp4"
        cls.music = cls.dir / "music.wav"
        make_click_track(cls.music)
        make_clips(cls.video, cls.tools)
        cls.track = analyze_track(cls.tools, str(cls.music))
        cls.shots = analyze_shots(str(cls.video), cls.tools.ffmpeg, fps=30.0, size=128)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_tempo(self) -> None:
        self.assertAlmostEqual(self.track.bpm, BPM, delta=3.0)
        expected = 9.0 / (60.0 / BPM)
        self.assertAlmostEqual(len(self.track.beats), expected, delta=2)

    def test_beats_even(self) -> None:
        diffs = np.diff(self.track.beats)
        self.assertLess(float(diffs.std()), 0.01, "сетка битов должна быть ровной")

    def test_downbeats(self) -> None:
        self.assertTrue(self.track.downbeats, "должны найтись сильные доли")
        spacing = np.diff(self.track.downbeats)
        self.assertAlmostEqual(float(np.median(spacing)), 4 * 60.0 / BPM, delta=0.05)

    def test_shots(self) -> None:
        self.assertGreaterEqual(len(self.shots), len(COLORS) - 1)
        for shot in self.shots:
            self.assertGreater(shot.duration, 0.3)

    def test_plan(self) -> None:
        plan = build_plan(self.shots, self.track, style="dynamic", target=5.0, fps=30)
        self.assertEqual(sum(s.frames for s in plan.segments), plan.total_frames)
        self.assertAlmostEqual(plan.duration, 5.0, delta=0.2)
        for a, b in zip(plan.segments, plan.segments[1:]):
            self.assertNotEqual(a.shot, b.shot, "соседние планы не должны повторять шот")
        for seg in plan.segments:
            self.assertGreaterEqual(seg.frames, 2)
            self.assertGreater(seg.src_dur, 0)
            self.assertGreaterEqual(seg.src_in, 0)

    def test_plan_styles(self) -> None:
        for style in ("dynamic", "cinematic", "slideshow"):
            plan = build_plan(self.shots, self.track, style=style, target=4.0, fps=30, seed=3)
            self.assertGreaterEqual(plan.cuts, 1, style)
            self.assertGreater(plan.duration, 3.5, style)

    def test_render(self) -> None:
        plan = build_plan(self.shots, self.track, style="dynamic", target=4.0, fps=30)
        out = self.dir / "out.mp4"
        result = render(plan, self.tools, str(out), self.dir / "work", workers=2)
        self.assertTrue(Path(result["output"]).is_file())
        self.assertGreater(Path(result["output"]).stat().st_size, 10_000)

        probe = subprocess.run(
            [self.tools.ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True,
        )
        frames = int(probe.stdout.strip())
        self.assertEqual(frames, plan.total_frames)

        audio = subprocess.run(
            [self.tools.ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True,
        )
        self.assertEqual(audio.stdout.strip(), "aac")


if __name__ == "__main__":
    unittest.main(verbosity=2)
