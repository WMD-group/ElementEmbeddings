from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "elementembeddings"
DOCS_ROOT = REPO_ROOT / "docs" / "python_api"

INTRO = """# ElementEmbeddings Python package

This API reference is generated from the public modules in `src/elementembeddings`.
The pages below stay aligned with the source tree automatically during docs builds
and prek runs.
""".strip()


def iter_public_modules() -> list[tuple[str, Path]]:
    modules: list[tuple[str, Path]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(SRC_ROOT)
        if any(part in {"tests", "__pycache__"} for part in relative.parts):
            continue
        if relative.name == "__init__.py":
            continue
        if relative.stem.startswith("_"):
            continue

        module = ".".join(("elementembeddings", *relative.with_suffix("").parts))
        doc_path = DOCS_ROOT / relative.with_suffix(".md")
        modules.append((module, doc_path))
    return modules


def write_module_pages(modules: list[tuple[str, Path]]) -> None:
    generated = {DOCS_ROOT / "index.md"}
    for module, doc_path in modules:
        generated.add(doc_path)
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        title = module.removeprefix("elementembeddings.")
        doc_path.write_text(f"# `{title}`\n\n::: {module}\n", encoding="utf-8")

    for existing in DOCS_ROOT.rglob("*.md"):
        if existing not in generated:
            existing.unlink()


def write_index(modules: list[tuple[str, Path]]) -> None:
    sections: dict[str, list[tuple[str, Path]]] = {}
    for module, doc_path in modules:
        package = module.removeprefix("elementembeddings.").split(".")[0]
        sections.setdefault(package, []).append((module, doc_path))

    lines = [INTRO, ""]
    for package in sorted(sections):
        lines.append(f"## `{package}`")
        lines.append("")
        for module, doc_path in sections[package]:
            label = module.removeprefix("elementembeddings.")
            relative_link = doc_path.relative_to(DOCS_ROOT).as_posix()
            lines.append(f"- [`{label}`]({relative_link})")
        lines.append("")

    (DOCS_ROOT / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    modules = iter_public_modules()
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    write_module_pages(modules)
    write_index(modules)


if __name__ == "__main__":
    main()
