#!/usr/bin/env python3
"""Audit EN vs KO translation parity in docs/.

Checks for:
1. EN pages with no KO counterpart (or vice versa).
2. Content length drift (>40% difference) between EN and KO versions.
3. KO pages with very low Hangul ratio (likely under-translated bodies),
   ignoring bibliographic pages.

Not wired to CI by default -- bibliographies and reports legitimately
use mostly English. Run on demand:

    python scripts/audit_translation_parity.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

HANGUL = re.compile(r"[가-힣]")
ASCII_LETTER = re.compile(r"[A-Za-z]")
SKIP_DIRS = ("/node_modules/", "/.vitepress/")


def content_lines(path: Path) -> int:
    text = path.read_text()
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5:]
    lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("<!--")]
    return len(lines)


def strip_for_lang(text: str) -> str:
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    return text


def main() -> int:
    en_files: list[Path] = []
    for p in sorted(DOCS.rglob("*.md")):
        rel = p.relative_to(REPO).as_posix()
        if any(s in f"/{rel}" for s in SKIP_DIRS):
            continue
        if "/ko/" in rel:
            continue
        en_files.append(p)

    missing: list[tuple[Path, Path]] = []
    drift: list[tuple[Path, Path, int, int, float]] = []

    for en in en_files:
        rel = en.relative_to(REPO).as_posix()
        ko_rel = rel.replace("docs/", "docs/ko/", 1)
        ko = REPO / ko_rel
        if not ko.exists():
            missing.append((en, ko))
            continue
        el = content_lines(en)
        kl = content_lines(ko)
        if el == 0:
            continue
        ratio = kl / el
        if ratio < 0.6 or ratio > 1.4:
            drift.append((en, ko, el, kl, ratio))

    under_translated: list[tuple[Path, float, int, int]] = []
    for ko in sorted(DOCS.glob("ko/**/*.md")):
        text = strip_for_lang(ko.read_text())
        h = len(HANGUL.findall(text))
        a = len(ASCII_LETTER.findall(text))
        if h + a < 200:
            continue  # stub
        ratio = h / (h + a)
        if ratio < 0.20:
            under_translated.append((ko, ratio, h, a))

    print(f"Checked {len(en_files)} EN markdown pages.\n")

    if missing:
        print(f"== Missing KO counterpart ({len(missing)}) ==")
        for en, ko in missing:
            print(f"  {en.relative_to(REPO)} -> expected at {ko.relative_to(REPO)}")
        print()
    if drift:
        print(f"== Content length drift ({len(drift)}) ==")
        for en, ko, el, kl, r in sorted(drift, key=lambda t: abs(1 - t[4]), reverse=True):
            print(f"  ratio={r:.2f}  {en.relative_to(REPO)} ({el}L) vs {ko.relative_to(REPO)} ({kl}L)")
        print()
    if under_translated:
        print(f"== KO pages with low Hangul ratio (<20%) ({len(under_translated)}) ==")
        print("  (bibliographies and reference lists are expected here)")
        for ko, r, h, a in sorted(under_translated):
            print(f"  hangul={r:.0%}  {ko.relative_to(REPO)} ({h} ko / {a} en chars)")
        print()

    if not (missing or drift or under_translated):
        print("All KO pages are in parity with EN.")
        return 0
    return 0  # informational only -- never fail


if __name__ == "__main__":
    sys.exit(main())
