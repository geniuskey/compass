#!/usr/bin/env python3
"""Lightweight dead-link scanner for docs/.

Scans every Markdown file under docs/ for absolute-path links of the form
`[text](/some/path)` and verifies that the target resolves to an existing
markdown file (or directory with index.md). Faster than a full VitePress
build for use as a pre-commit / quick-feedback check.

Limitations vs. `vitepress build`:
- Only checks links that start with `/`. Relative links and external URLs
  are skipped.
- Does not verify anchor fragments (`#section-id`).
- Does not check Vue component `:href` bindings.

Usage:
    python scripts/check_doc_links.py            # scan, exit 1 on errors
    python scripts/check_doc_links.py --quiet    # only print failures
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\((/[^)\s]+)\)")


def resolve(target: str) -> Path | None:
    """Resolve `/foo/bar` (with or without `.md`/anchor) to a real file."""
    # strip anchor
    target = target.split("#", 1)[0]
    if not target:
        return None
    # /compass/... is the deployed base; map back to docs root
    if target.startswith("/compass/"):
        target = target[len("/compass"):]
    rel = target.lstrip("/")
    candidates = [
        DOCS / rel,
        DOCS / f"{rel}.md",
        DOCS / rel / "index.md",
    ]
    if rel.endswith("/"):
        candidates.append(DOCS / f"{rel}index.md")
    for c in candidates:
        if c.exists():
            return c
    return None


SKIP_DIRS = {"node_modules", ".vitepress/cache", ".vitepress/dist"}


def _iter_md(root: Path):
    for f in root.rglob("*.md"):
        rel = f.relative_to(root).as_posix()
        if any(rel.startswith(s) or f"/{s}/" in f"/{rel}" for s in SKIP_DIRS):
            continue
        yield f


def scan(quiet: bool) -> int:
    errors: list[tuple[Path, int, str, str]] = []
    files = list(_iter_md(DOCS))
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in LINK_RE.finditer(line):
                text, target = m.group(1), m.group(2)
                # Skip pure asset paths handled by VitePress (e.g., images
                # under /public are referenced without `/public` prefix; let
                # vitepress build catch those).
                if target.startswith(("/images/", "/assets/", "/public/")):
                    continue
                if resolve(target) is None:
                    errors.append((f, i, text, target))

    if errors:
        for f, i, text, target in errors:
            rel = f.relative_to(REPO)
            print(f"{rel}:{i}: dead link [{text}]({target})")
        print(f"\n{len(errors)} dead link(s) in {len(files)} markdown files.")
        return 1
    if not quiet:
        print(f"OK: scanned {len(files)} markdown files, no dead absolute links.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    return scan(args.quiet)


if __name__ == "__main__":
    sys.exit(main())
