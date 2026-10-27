"""Поиск внешних утилит (ffmpeg/ffprobe) и окружения.

Скилл должен работать без настройки: если пользователь поставил ffmpeg
любым из популярных способов, мы его найдём. Если не поставил — скажем
человеческим текстом, что сделать.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path

# Типичные места установки на Windows, которые не попадают в PATH
# у уже запущенного процесса (winget пишет в переменную пути, но она
# подхватывается только новым терминалом).
WIN_GLOBS = [
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg*\bin\ffmpeg.exe",
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\*\ffmpeg*\bin\ffmpeg.exe",
    r"%USERPROFILE%\scoop\shims\ffmpeg.exe",
    r"%ProgramData%\chocolatey\bin\ffmpeg.exe",
    r"%ProgramFiles%\ffmpeg\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
]

POSIX_GLOBS = [
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
    "/snap/bin/ffmpeg",
]


@dataclass
class Tools:
    ffmpeg: str
    ffprobe: str
    version: str = ""

    def describe(self) -> str:
        return f"ffmpeg {self.version} ({self.ffmpeg})"


class ToolNotFound(RuntimeError):
    pass


def _find_pair(candidate_ffmpeg: str) -> Tools | None:
    """По пути к ffmpeg находим ffprobe рядом с ним."""
    p = Path(candidate_ffmpeg)
    if not p.is_file():
        return None
    probe = p.with_name(p.name.replace("ffmpeg", "ffprobe"))
    if not probe.is_file():
        probe = p.with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    if not probe.is_file():
        probe = p
    return Tools(str(p), str(probe))


def _version(binary: str) -> str:
    try:
        out = subprocess.run(
            [binary, "-version"], capture_output=True, text=True, errors="replace", timeout=20
        ).stdout
        m = re.search(r"ffmpeg version (\S+)", out)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def find_tools(strict: bool = True) -> Tools:
    """Ищет ffmpeg/ffprobe: переменная окружения → PATH → типовые пути → wheel."""
    env_ffmpeg = os.environ.get("MONTAZH_FFMPEG") or os.environ.get("FFMPEG_BINARY")
    if env_ffmpeg:
        found = _find_pair(env_ffmpeg)
        if found:
            found.version = _version(found.ffmpeg)
            return found

    on_path = shutil.which("ffmpeg")
    if on_path:
        found = _find_pair(on_path)
        if found:
            found.version = _version(found.ffmpeg)
            return found

    patterns = WIN_GLOBS if sys.platform == "win32" else POSIX_GLOBS
    for pattern in patterns:
        for path in sorted(glob(os.path.expandvars(pattern)), reverse=True):
            found = _find_pair(path)
            if found:
                found.version = _version(found.ffmpeg)
                return found

    # Последний шанс: колесо imageio-ffmpeg приносит статический бинарник.
    try:
        import imageio_ffmpeg  # type: ignore

        binary = imageio_ffmpeg.get_ffmpeg_exe()
        found = _find_pair(binary)
        if found:
            found.version = _version(found.ffmpeg)
            return found
    except Exception:
        pass

    if strict:
        raise ToolNotFound(
            "Не нашёл ffmpeg. Установите одним из способов:\n"
            "  winget install Gyan.FFmpeg      (Windows)\n"
            "  brew install ffmpeg             (macOS)\n"
            "  sudo apt install ffmpeg         (Debian/Ubuntu)\n"
            "или укажите путь: set MONTAZH_FFMPEG=C:\\ffmpeg\\bin\\ffmpeg.exe"
        )
    raise ToolNotFound("ffmpeg не найден")


def run_bytes(cmd: list[str], *, check: bool = True) -> bytes:
    """Запуск команды, которая отдаёт двоичные данные в stdout (PCM-поток)."""
    proc = subprocess.run(cmd, capture_output=True)
    if check and proc.returncode != 0:
        tail = (proc.stderr or b"")[-3000:].decode("utf-8", "replace")
        raise RuntimeError(f"Команда упала ({proc.returncode}):\n{' '.join(cmd[:12])}...\n{tail}")
    return proc.stdout or b""


def run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Запуск внешней команды с понятной ошибкой при падении."""
    proc = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        errors="replace",
        encoding="utf-8" if capture else None,
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or "")[-3000:]
        raise RuntimeError(f"Команда упала ({proc.returncode}):\n{' '.join(cmd[:12])}...\n{tail}")
    return proc
