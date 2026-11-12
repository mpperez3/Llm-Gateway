"""Integridad de la memoria del proyecto y de las skills para agentes.

Estos ficheros son instrucciones para agentes, no documentación decorativa: si se
desincronizan, un agente tomará decisiones equivocadas. Este test es el guardián.

Se ejecuta con `uv run pytest tests/test_project_memory.py -q`.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / ".opencode" / "memory"
SKILLS = ROOT / ".opencode" / "skills"
AGENTS = ROOT / "AGENTS.md"

REQUIRED_MEMORY = {
    "INDEX.md": "index",
    "invariants.md": "invariants",
    "preferences.md": "preferences",
    "decisions.md": "decisions",
    "events.md": "events",
    "known-issues.md": "known_issues",
    "glossary.md": "glossary",
    "skills.index.md": "skills_index",
}

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
EVENT_RE = re.compile(r"^- \*\*(\d{4}-\d{2}-\d{2})\*\* · (.+)$", re.MULTILINE)


def _frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    assert match, f"{path.relative_to(ROOT)}: falta frontmatter YAML delimitado por ---"
    data = yaml.safe_load(match.group(1))
    assert isinstance(data, dict), f"{path.relative_to(ROOT)}: frontmatter no es un mapping"
    return data, text[match.end():]


def test_memory_files_exist_with_valid_frontmatter():
    assert MEMORY.is_dir(), ".opencode/memory/ no existe"

    for filename, memory_id in REQUIRED_MEMORY.items():
        path = MEMORY / filename
        assert path.is_file(), f"falta {path.relative_to(ROOT)}"
        meta, body = _frontmatter(path)

        assert meta.get("memory_file") == memory_id, (
            f"{filename}: memory_file debe ser {memory_id!r}, es {meta.get('memory_file')!r}"
        )
        assert meta.get("version") == 1, f"{filename}: version debe ser 1"
        assert meta.get("load") in {"always", "on_demand"}, (
            f"{filename}: load debe ser 'always' u 'on_demand'"
        )

        updated = meta.get("updated")
        assert isinstance(updated, (dt.date, dt.datetime)), f"{filename}: updated ausente o inválido"
        assert updated <= dt.date.today(), f"{filename}: updated está en el futuro"

        assert body.lstrip().startswith("# "), f"{filename}: el cuerpo debe empezar por un H1"


def test_always_loaded_memory_files_are_referenced_by_agents_md():
    """Si un fichero se inyecta siempre, AGENTS.md no puede contradecirlo ni ignorarlo."""
    agents = AGENTS.read_text(encoding="utf-8")
    for filename in ("invariants.md", "preferences.md", "skills.index.md", "glossary.md"):
        assert filename in agents, f"AGENTS.md no referencia .opencode/memory/{filename}"


def test_invariants_are_numbered_and_non_empty():
    text = (MEMORY / "invariants.md").read_text(encoding="utf-8")
    numbers = [int(n) for n in re.findall(r"^(\d+)\.\s+\*\*", text, re.MULTILINE)]
    assert numbers, "invariants.md no contiene invariantes numerados"
    assert numbers == list(range(1, len(numbers) + 1)), f"numeración no consecutiva: {numbers}"

    # Un invariante sin motivo/fecha no sirve para decidir.
    entries = re.split(r"^\d+\.\s+\*\*", text, flags=re.MULTILINE)[1:]
    for index, entry in enumerate(entries, start=1):
        assert "Origen:" in entry or "origen:" in entry, (
            f"invariante {index}: falta la línea 'Origen:' (motivo/fecha)"
        )


def test_skill_index_lists_every_skill():
    """El índice siempre inyectado debe cubrir exactamente las skills de disco."""
    text = (MEMORY / "skills.index.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\|\s*`([a-z0-9-]+)`\s*\|", text, re.MULTILINE))
    on_disk = {p.name for p in SKILLS.iterdir() if p.is_dir()}

    assert listed, "skills.index.md no tiene tabla de skills"
    assert listed == on_disk, (
        f"skills.index.md y disco difieren. Solo en índice: {sorted(listed - on_disk)}; "
        f"solo en disco: {sorted(on_disk - listed)}"
    )
    assert "lee el `SKILL.md` completo" in text or "lee el SKILL.md completo" in text, (
        "skills.index.md debe ordenar leer la skill completa antes de actuar"
    )


def test_skill_descriptions_are_distinct():
    """Dos descripciones casi iguales hacen que el modelo elija mal la skill."""
    seen: dict[str, str] = {}
    for skill in SKILLS.glob("*/SKILL.md"):
        meta, _ = _frontmatter(skill)
        key = meta["description"].strip().lower()[:90]
        assert key not in seen, (
            f"{skill.parent.name} y {seen[key]} tienen descripciones casi idénticas"
        )
        seen[key] = skill.parent.name


def test_events_month_headers_match_entry_dates():
    """Una entrada de septiembre bajo el header de octubre se pierde al leer."""
    current: str | None = None
    for line in (MEMORY / "events.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            header = line[3:].strip()
            if "anteriores" in header:
                current = "older"
                continue
            match = re.match(r"(\d{4})-(\d{2})", header)
            current = f"{match.group(1)}-{match.group(2)}" if match else "older"
            continue
        entry = EVENT_RE.match(line)
        if not entry or current is None:
            continue
        month = entry.group(1)[:7]
        if current == "older":
            assert month <= "2026-08", f"entrada {entry.group(1)} bajo un header de archivo"
        else:
            assert month == current, (
                f"entrada {entry.group(1)} bajo el header {current}"
            )


def test_events_are_chronological_and_bounded():
    text = (MEMORY / "events.md").read_text(encoding="utf-8")
    entries = EVENT_RE.findall(text)
    assert entries, "events.md no tiene entradas con formato '- **YYYY-MM-DD** · …'"

    dates = [dt.date.fromisoformat(date) for date, _ in entries]
    assert dates == sorted(dates, reverse=True), "events.md debe estar ordenado de más reciente a más antiguo"
    assert len(entries) <= 60, (
        f"events.md tiene {len(entries)} entradas (máx 60): mueve las antiguas a events-<año>.md"
    )


def test_every_skill_has_valid_frontmatter():
    assert SKILLS.is_dir(), ".opencode/skills/ no existe"
    skill_dirs = sorted(p for p in SKILLS.iterdir() if p.is_dir() and not p.name.startswith("."))
    assert skill_dirs, ".opencode/skills/ está vacío"

    for directory in skill_dirs:
        path = directory / "SKILL.md"
        assert path.is_file(), f"{directory.name}: falta SKILL.md"
        meta, body = _frontmatter(path)

        name = meta.get("name")
        assert name == directory.name, (
            f"{directory.name}: frontmatter name={name!r} no coincide con el directorio"
        )
        assert SKILL_NAME_RE.match(name), f"{directory.name}: nombre no kebab-case"
        assert len(name) <= 64, f"{directory.name}: nombre > 64 caracteres"

        description = meta.get("description")
        assert isinstance(description, str) and len(description) >= 40, (
            f"{directory.name}: description ausente o demasiado corta para disparar la skill"
        )
        assert "úsala" in description.lower() or "usa esta skill" in description.lower() or "use this" in description.lower(), (
            f"{directory.name}: la description debe decir cuándo usarla (cláusula 'úsala siempre que…')"
        )
        assert len(body.strip()) > 0, f"{directory.name}: cuerpo vacío"


def test_skill_list_in_agents_md_matches_directories():
    agents = AGENTS.read_text(encoding="utf-8")
    section = agents.split("## 6. Skills del proyecto", 1)
    assert len(section) == 2, "AGENTS.md no tiene la sección de skills"
    body = section[1].split("\n## ", 1)[0]
    referenced = set(re.findall(r"^\|\s*`([a-z0-9-]+)`\s*\|", body, re.MULTILINE))
    on_disk = {p.name for p in SKILLS.iterdir() if p.is_dir()}

    missing_dirs = referenced - on_disk
    missing_refs = on_disk - referenced
    assert not missing_dirs, f"AGENTS.md cita skills que no existen: {sorted(missing_dirs)}"
    assert not missing_refs, f"hay skills sin entrada en AGENTS.md: {sorted(missing_refs)}"


def test_no_legacy_product_name_in_agent_facing_files():
    """`heimdall-gateway` solo puede aparecer como alias deprecado explícito.

    `events.md` y `known-issues.md` son históricos: ahí el nombre antiguo es evidencia, no
    una instrucción. El resto de recursos de agente debe usar el nombre actual.
    """
    legacy = re.compile(r"heimdall[\s-]gateway", re.IGNORECASE)
    actionable = [
        *SKILLS.rglob("*.md"),
        MEMORY / "INDEX.md",
        MEMORY / "invariants.md",
        MEMORY / "preferences.md",
        MEMORY / "decisions.md",
    ]
    offenders = [str(p.relative_to(ROOT)) for p in actionable if legacy.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"nomenclatura antigua en recursos de agente: {offenders}"

    agents = AGENTS.read_text(encoding="utf-8")
    legacy = [line for line in agents.splitlines() if "heimdall" in line.lower()]
    assert len(legacy) <= 1, "AGENTS.md menciona 'heimdall' más de una vez"
    for line in legacy:
        assert "deprecad" in line.lower() or "alias" in line.lower(), (
            "la única mención a heimdall en AGENTS.md debe ser como alias deprecado"
        )


def test_project_opencode_config_forces_always_loaded_memory():
    """El proyecto debe inyectar la memoria crítica por config nativa, no solo por goodwill."""
    config_path = ROOT / "opencode.json"
    assert config_path.is_file(), "falta opencode.json del proyecto"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    instructions = config.get("instructions")
    assert isinstance(instructions, list) and instructions, "opencode.json sin 'instructions'"
    assert ".opencode/memory/invariants.md" in instructions, (
        "invariants.md debe inyectarse siempre vía opencode.json"
    )
    for entry in instructions:
        assert (ROOT / entry).is_file(), f"opencode.json referencia {entry}, que no existe"


def test_no_second_skill_tree_with_stale_copies():
    """Una skill duplicada en otra ruta se descubre y compite con la canónica."""
    for candidate in (ROOT / ".agents" / "skills", ROOT / ".claude" / "skills", ROOT / "skills"):
        if not candidate.is_dir():
            continue
        for skill_md in candidate.glob("*/SKILL.md"):
            name = skill_md.parent.name
            canonical = SKILLS / name / "SKILL.md"
            assert not canonical.exists(), (
                f"{skill_md.relative_to(ROOT)} duplica la skill canónica {canonical.relative_to(ROOT)}"
            )


def test_agents_md_keeps_non_negotiables_visible():
    """AGENTS.md es el único canal garantizado: los no-negociables deben estar inline."""
    text = AGENTS.read_text(encoding="utf-8")
    section = text.split("## 2. No negociables", 1)
    assert len(section) == 2, "AGENTS.md perdió la sección de no negociables"
    body = section[1].split("\n## ", 1)[0]
    numbered = re.findall(r"^\d+\.\s+\*\*", body, re.MULTILINE)
    assert len(numbered) >= 10, (
        f"solo {len(numbered)} no-negociables inline; el resumen debe mantenerse aunque "
        "el detalle viva en invariants.md"
    )


def test_agents_md_stays_a_router():
    lines = AGENTS.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 260, (
        f"AGENTS.md tiene {len(lines)} líneas: se inyecta en cada sesión. "
        "Mueve detalle a .opencode/memory/ o docs/."
    )


@pytest.mark.parametrize("path_text", ["AGENTS.md", ".opencode/memory/INDEX.md"])
def test_referenced_memory_paths_exist(path_text: str):
    path = ROOT / path_text
    text = path.read_text(encoding="utf-8")
    for referenced in re.findall(r"`(\.opencode/memory/[A-Za-z0-9_.-]+\.md)`", text):
        assert (ROOT / referenced).is_file(), f"{path_text} referencia {referenced}, que no existe"


def test_skill_relative_links_resolve():
    for skill in SKILLS.glob("*/SKILL.md"):
        text = skill.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?://)([^)#]+)\)", text):
            resolved = (skill.parent / target).resolve()
            assert resolved.exists(), f"{skill.relative_to(ROOT)}: enlace roto -> {target}"
