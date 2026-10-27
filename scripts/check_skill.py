"""Проверка, что SKILL.md скилла читается так же, как у остальных скиллов DSH:
YAML-заголовок между ---, ключи name / description / whenToUse, без двоеточий
в значениях (их не все парсеры переживают).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(path: Path, require_dir_name: bool = True) -> int:
    text = path.read_text("utf-8")
    if not text.startswith("---\n"):
        print(f"✗ {path.name}: нет YAML-заголовка")
        return 1
    end = text.find("\n---", 4)
    if end == -1:
        print(f"✗ {path.name}: заголовок не закрыт")
        return 1
    header = text[4:end]
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if not line.strip():
            continue
        m = re.match(r"^([a-zA-Z_][\w-]*): (.+)$", line)
        if not m:
            print(f"✗ {path.name}: строка заголовка не разобрана: {line[:60]!r}")
            return 1
        key, value = m.group(1), m.group(2)
        if ": " in value:
            print(f"✗ {path.name}: в поле {key} есть «: » — часть парсеров на этом ломается")
            return 1
        fields[key] = value
    missing = {"name", "description", "whenToUse"} - set(fields)
    if missing:
        print(f"✗ {path.name}: нет полей {', '.join(sorted(missing))}")
        return 1
    if require_dir_name and fields["name"] != path.parent.name:
        print(f"✗ {path.name}: name={fields['name']!r} не совпадает с папкой {path.parent.name!r}")
        return 1
    body = text[end + 4:]
    if len(fields["description"]) > 500:
        print(f"✗ {path.name}: description длиннее 500 символов (каталог обрежет)")
        return 1
    if not body.strip():
        print(f"✗ {path.name}: пустое тело")
        return 1
    print(f"✓ {path.name}: {fields['name']}, description {len(fields['description'])} симв., "
          f"тело {len(body.splitlines())} строк")
    return 0


def main() -> int:
    code = check(ROOT / "SKILL.md", require_dir_name=False)
    skills_dir = Path.home() / ".dsh" / "skills"
    if skills_dir.is_dir():
        for folder in sorted(p for p in skills_dir.iterdir() if (p / "SKILL.md").is_file()):
            code |= check(folder / "SKILL.md", require_dir_name=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
