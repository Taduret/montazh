"""Командная строка скилла.

    python -m montazh doctor
    python -m montazh auto  --src "На монтаж/Тест 1" --style dynamic --duration 95 --aspect 1:1
    python -m montazh auto  --src ... --music track.mp3 --style cinematic --duration 60 --preview

Всё остальное (анализ, планирование, рендер) — отдельные подкоманды, чтобы
можно было пересобирать только нужный шаг.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .audio import TrackAnalysis, analyze_track
from .env import ToolNotFound, find_tools
from .media import Media, collect_sources, probe
from .plan import STYLES, build_plan, summarize_plan
from .render import contact_sheet, poster, render
from .shots import Shot, analyze_shots, shots_from_entries, summarize

DEFAULT_WORK = ".montazh"


def log(msg: str) -> None:
    print(msg, flush=True)


def _cache_path(work: Path, kind: str, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return work / "cache" / f"{kind}_{digest}.json"


def _pick_track(tracks: list[Media], music: str | None) -> Media:
    if music:
        return probe(find_tools(), music) if Path(music).is_file() else None  # type: ignore[return-value]
    if not tracks:
        raise SystemExit(
            "В папке нет ни одного аудиофайла. Укажите трек: --music путь\\к\\треку.mp3"
        )
    return max(tracks, key=lambda m: m.duration)


def analyze_sources(clips: list[Media], track: Media, work: Path, *,
                    music_start: float | None = None, duration: float | None = None,
                    detect_fps: float = 30.0, force: bool = False) -> tuple[TrackAnalysis, list[Shot]]:
    tools = find_tools()
    key = f"{track.path}|{track.duration}|{music_start}|{duration}"
    track_cache = _cache_path(work, "track", key)
    track_analysis: TrackAnalysis
    if track_cache.exists() and not force:
        data = json.loads(track_cache.read_text("utf-8"))
        track_analysis = TrackAnalysis(**data)
        log(f"  музыка: кеш ({track_analysis.bpm} BPM, {len(track_analysis.beats)} битов)")
    else:
        log(f"  музыка: анализ {Path(track.path).name} …")
        t0 = time.time()
        track_analysis = analyze_track(tools, track.path, start=music_start, duration=duration)
        track_cache.parent.mkdir(parents=True, exist_ok=True)
        track_analysis.save(track_cache)
        log(f"  музыка: {track_analysis.bpm} BPM, {len(track_analysis.beats)} битов, "
            f"сетка {track_analysis.grid}, {time.time() - t0:.1f} c")

    shots: list[Shot] = []
    for clip in clips:
        key = f"{clip.path}|{clip.duration}|{detect_fps}"
        cache = _cache_path(work, "shots", key)
        if cache.exists() and not force:
            data = json.loads(cache.read_text("utf-8"))
            part = shots_from_entries(data["shots"])
            log(f"  {Path(clip.path).name}: кеш, {len(part)} шотов")
        else:
            log(f"  {Path(clip.path).name}: ищу склейки …")
            t0 = time.time()
            part = analyze_shots(clip.path, tools.ffmpeg, fps=detect_fps)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps({"shots": [s.to_dict() for s in part]}, ensure_ascii=False), "utf-8"
            )
            log(f"  {Path(clip.path).name}: {len(part)} шотов, {time.time() - t0:.1f} c")
        offset = len(shots)
        for s in part:
            s.index = offset + s.index
            shots.append(s)
    return track_analysis, shots


def cmd_doctor(args) -> int:
    log(f"montazh {__version__}")
    try:
        tools = find_tools()
    except ToolNotFound as exc:
        log(f"✗ {exc}")
        return 2
    log(f"✓ {tools.describe()}")
    try:
        from .env import run

        out = run([tools.ffprobe, "-version"]).stdout.splitlines()[0]
        log(f"✓ {out}")
    except Exception as exc:  # noqa: BLE001
        log(f"✗ ffprobe: {exc}")
        return 2
    log(f"✓ python {sys.version.split()[0]}")
    try:
        import numpy

        log(f"✓ numpy {numpy.__version__}")
    except Exception:
        log("✗ numpy не установлен — pip install numpy")
        return 2
    log(f"✓ ядер: {os.cpu_count()}")
    return 0


def _build(args, work: Path):
    tools = find_tools()
    clips, tracks = collect_sources(tools, args.src, recursive=not args.no_recursive)
    if not clips:
        raise SystemExit(f"В папке {args.src} не нашёл ни одного видео или фото")
    track = _pick_track(tracks, args.music)
    if track is None:
        raise SystemExit(f"Не нашёл трек: {args.music}")

    log(f"Материал: {len(clips)} файлов, {sum(c.duration for c in clips):.1f} c")
    log(f"Трек: {Path(track.path).name} ({track.duration:.1f} c)")

    track_analysis, shots = analyze_sources(
        clips, track, work,
        music_start=args.music_start, duration=None,
        detect_fps=args.detect_fps, force=args.force,
    )
    sizes = [(c.width, c.height) for c in clips if c.width]
    plan = build_plan(
        shots, track_analysis,
        style=args.style, target=args.duration, fps=args.fps,
        aspect=args.aspect, height=args.height,
        music_start=args.music_start, seed=args.seed, sizes=sizes,
    )
    log("")
    log(f"План: {plan.cuts} склеек, {plan.duration:.1f} c, "
        f"{plan.width}×{plan.height}, {plan.fps} fps, стиль {plan.style}")
    for note in plan.notes:
        log(f"  · {note}")
    return tools, clips, track_analysis, shots, plan


def cmd_auto(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    tools, clips, track_analysis, shots, plan = _build(args, work)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = work / "render"

    if args.preview:
        plan.width = int(plan.width * 0.55) // 2 * 2
        plan.height = int(plan.height * 0.55) // 2 * 2
        log(f"  (черновик: {plan.width}×{plan.height})")
    plan.save(work / "plan.json")
    (work / "shots.json").write_text(
        json.dumps([s.to_dict() for s in shots], ensure_ascii=False, indent=2), "utf-8"
    )
    track_analysis.save(work / "track.json")

    log("")
    log("Рендер сегментов …")
    t0 = time.time()
    result = render(plan, tools, str(out_path), work_dir, workers=args.workers,
                    on_progress=lambda done, total: log(f"  {done}/{total}"))
    elapsed = time.time() - t0

    poster_path = str(Path(args.out).with_suffix(".jpg"))
    poster(plan, tools, result["output"], poster_path)
    sheet = str(Path(args.out).with_name(Path(args.out).stem + "_sheet.jpg"))
    contact_sheet(plan, tools, result["output"], sheet)

    final = probe(tools, result["output"])
    report = {
        "output": str(out_path),
        "poster": poster_path,
        "sheet": sheet,
        "media": {
            "duration": round(final.duration, 3),
            "size": f"{final.width}x{final.height}",
            "fps": round(final.fps, 2),
            "has_audio": final.has_audio,
            "bytes": os.path.getsize(out_path),
        },
        "plan": summarize_plan(plan),
        "track": {
            "bpm": track_analysis.bpm,
            "beats": len(track_analysis.beats),
            "downbeats": len(track_analysis.downbeats),
            "grid": track_analysis.grid,
            "sections": track_analysis.sections,
            "drop": track_analysis.drop,
        },
        "shots": summarize(shots),
        "render_seconds": round(elapsed, 1),
    }
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    _write_report_md(Path(args.out).with_suffix(".md"), report, plan)

    log("")
    log(f"Готово за {elapsed:.0f} c: {out_path}")
    log(f"  {final.width}×{final.height}, {final.duration:.1f} c, {final.fps:.0f} fps, "
        f"звук {'есть' if final.has_audio else 'нет'}")
    log(f"  постер: {poster_path}")
    log(f"  нарезка кадров: {sheet}")
    return 0


def _write_report_md(path: Path, report: dict, plan) -> None:
    p = report["plan"]
    t = report["track"]
    s = report["shots"]
    lines = [
        "# Отчёт о монтаже",
        "",
        f"**Файл:** `{report['output']}`",
        f"**Длительность:** {report['media']['duration']:.1f} c · "
        f"{report['media']['size']} · {report['media']['fps']:.0f} fps",
        "",
        "## Музыка",
        f"- BPM: {t['bpm']} ({t['grid']})",
        f"- Битов: {t['beats']}, долей: {t['downbeats']}",
        f"- Секции: " + ", ".join(f"{x['start']:.0f}–{x['end']:.0f} c ({x['level']})" for x in t["sections"][:6]),
        "",
        "## Монтаж",
        f"- Стиль: {p['style']}",
        f"- Склеек: {p['cuts']} ({p['cuts_per_sec']} в секунду), средний план {p['avg_shot']} c",
        f"- Уникальных шотов: {p['unique_shots']} из {s['count']} найденных",
        f"- Фрагмент трека: с {p['music_start']:.1f} c, {p['duration']:.1f} c",
        "",
    ]
    if p["notes"]:
        lines += ["## Замечания"] + [f"- {n}" for n in p["notes"]] + [""]
    lines += [
        "## Исходники",
        f"- Файлов: {s['sources']}, шотов: {s['count']}",
        f"- Длительности шотов: медиана {s['median']} c, мин {s['min']}, макс {s['max']}",
        f"- Тёмных шотов: {s['dark']}",
        "",
        f"![Постер]({Path(report['poster']).name})",
        "",
    ]
    path.write_text("\n".join(lines), "utf-8")


def cmd_analyze(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    tools, clips, track_analysis, shots, plan = _build(args, work)
    log("")
    log(f"Шотов: {len(shots)}, суммарно {sum(s.duration for s in shots):.1f} c")
    log(f"План сохранится в {work / 'plan.json'} при рендере")
    return 0


def cmd_preview(args) -> int:
    args.preview = True
    args.duration = args.duration or 20
    args.height = args.height or 480
    return cmd_auto(args)


def cmd_variants(args) -> int:
    """Три стиля на одном материале — чтобы выбрать на глаз."""
    styles = args.styles.split(",") if args.styles else ["dynamic", "cinematic", "slideshow"]
    base = Path(args.out)
    original_out = args.out
    for style in styles:
        style = style.strip()
        if style not in STYLES:
            raise SystemExit(f"Неизвестный стиль {style!r}. Доступны: {', '.join(STYLES)}")
        args.style = style
        args.out = str(base.with_name(f"{base.stem}_{style}{base.suffix}"))
        log("")
        log(f"=== {style} ===")
        code = cmd_auto(args)
        if code:
            return code
    args.out = original_out
    log("")
    log(f"Готово: {len(styles)} вариантов рядом с {base.parent}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="montazh",
        description="Монтаж видео под музыку: биты → шоты → EDL → mp4",
    )
    parser.add_argument("--version", action="version", version=f"montazh {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, src_required: bool = True) -> None:
        p.add_argument("--src", required=src_required, help="папка с клипами и музыкой")
        p.add_argument("--music", help="явный путь к треку")
        p.add_argument("--out", default="out/montazh.mp4", help="куда положить результат")
        p.add_argument("--work", default=DEFAULT_WORK, help="рабочая папка (кеш, план)")
        p.add_argument("--style", default="dynamic", choices=list(STYLES), help="стиль монтажа")
        p.add_argument("--duration", type=float, default=None, help="длительность ролика, c")
        p.add_argument("--aspect", default="source", choices=["source", "1:1", "9:16", "16:9", "4:5"])
        p.add_argument("--height", type=int, default=None, help="высота кадра, px")
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--music-start", type=float, default=None, help="с какой секунды трека")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=None)
        p.add_argument("--detect-fps", type=float, default=30.0, help="fps анализа склеек")
        p.add_argument("--no-recursive", action="store_true")
        p.add_argument("--force", action="store_true", help="пересчитать кеш анализа")
        p.add_argument("--preview", action="store_true", help="быстрый низкий рендер")

    auto = sub.add_parser("auto", help="анализ + план + рендер одной командой")
    common(auto)
    auto.set_defaults(func=cmd_auto)

    analyze = sub.add_parser("analyze", help="только анализ и план")
    common(analyze)
    analyze.set_defaults(func=cmd_analyze)

    preview = sub.add_parser("preview", help="короткий черновик для проверки")
    common(preview)
    preview.set_defaults(func=cmd_preview)

    variants = sub.add_parser("variants", help="все три стиля на одном материале")
    common(variants)
    variants.add_argument("--styles", default="dynamic,cinematic,slideshow")
    variants.set_defaults(func=cmd_variants)

    doctor = sub.add_parser("doctor", help="проверить ffmpeg и зависимости")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ToolNotFound, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            raise
        log(f"✗ {exc}")
        return 2
    except KeyboardInterrupt:
        log("Прервано")
        return 130
    except Exception as exc:  # noqa: BLE001
        log(f"✗ {exc}")
        return 1
