#!/usr/bin/env python3
"""Single source-of-truth for solver tables in docs and CLAUDE.md.

Edit `SOLVERS` below and run this script to regenerate all solver tables.
Each table region in target files is delimited by:

    <!-- solver-table-start -->
    ...generated content...
    <!-- solver-table-end -->

Run:
    python scripts/sync_solver_tables.py
    python scripts/sync_solver_tables.py --check   # exit 1 if drift
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SOLVERS = [
    {
        "name": "torcwa",
        "type": "RCWA",
        "module": "compass.solvers.rcwa.torcwa_solver",
        "framework": "PyTorch",
        "notes_en": "PyTorch S-matrix RCWA. Default backend; TF32 disabled for stability.",
        "notes_ko": "PyTorch S-행렬 RCWA. 기본 백엔드이며 안정성을 위해 TF32 비활성화.",
    },
    {
        "name": "grcwa",
        "type": "RCWA",
        "module": "compass.solvers.rcwa.grcwa_solver",
        "framework": "JAX/NumPy",
        "notes_en": "JAX/NumPy RCWA. Cross-validation reference. **Critical** -- never remove.",
        "notes_ko": "JAX/NumPy RCWA. 교차 검증 기준점. **핵심** -- 제거 금지.",
    },
    {
        "name": "meent",
        "type": "RCWA",
        "module": "compass.solvers.rcwa.meent_solver",
        "framework": "PyTorch/JAX/NumPy",
        "notes_en": "Multi-backend RCWA with analytic eigendecomposition.",
        "notes_ko": "다중 백엔드 RCWA, 해석적 고유값 분해 지원.",
    },
    {
        "name": "fmmax",
        "type": "RCWA",
        "module": "compass.solvers.rcwa.fmmax_solver",
        "framework": "JAX",
        "notes_en": "JAX FMM with 4 selectable vector formulations.",
        "notes_ko": "4가지 정식화 선택 가능한 JAX 벡터 FMM.",
    },
    {
        "name": "fdtd_flaport",
        "type": "FDTD",
        "module": "compass.solvers.fdtd.flaport_solver",
        "framework": "PyTorch",
        "notes_en": "PyTorch 2.5D FDTD, GPU + autograd.",
        "notes_ko": "PyTorch 2.5D FDTD, GPU + autograd.",
    },
    {
        "name": "fdtdz",
        "type": "FDTD",
        "module": "compass.solvers.fdtd.fdtdz_solver",
        "framework": "JAX",
        "notes_en": "JAX 2D FDTD (z-invariant); fast for 2D cross-sections.",
        "notes_ko": "JAX 2D FDTD (z-불변); 2D 단면에 매우 빠름.",
    },
    {
        "name": "fdtdx",
        "type": "FDTD",
        "module": "compass.solvers.fdtd.fdtdx_solver",
        "framework": "JAX",
        "notes_en": "JAX 3D FDTD, multi-GPU, fully differentiable, MIT license.",
        "notes_ko": "JAX 3D FDTD, 멀티 GPU, 완전 미분 가능, MIT 라이선스.",
    },
    {
        "name": "meep",
        "type": "FDTD",
        "module": "compass.solvers.fdtd.meep_solver",
        "framework": "C++/Python",
        "notes_en": "C++/Python 3D FDTD with subpixel averaging and adjoint gradients.",
        "notes_ko": "C++/Python 3D FDTD, 서브픽셀 평균화 및 수반(adjoint) 그래디언트 지원.",
    },
    {
        "name": "tmm",
        "type": "TMM",
        "module": "compass.solvers.tmm.tmm_solver",
        "framework": "NumPy",
        "notes_en": "1D planar stacks only, ~1000x faster than RCWA.",
        "notes_ko": "1D 평면 스택 전용, RCWA 대비 ~1000배 빠름.",
    },
]

MARK_START = "<!-- solver-table-start -->"
MARK_END = "<!-- solver-table-end -->"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---" for _ in headers]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(f" {s} " for s in sep) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_en_full() -> str:
    rows = [
        [f"`{s['name']}`", f"`{s['module']}`", s["type"], s["notes_en"]]
        for s in SOLVERS
    ]
    return render_table(["Name", "Module", "Type", "Notes"], rows)


def render_ko_full() -> str:
    rows = [
        [f"`{s['name']}`", f"`{s['module']}`", s["type"], s["notes_ko"]]
        for s in SOLVERS
    ]
    return render_table(["이름", "모듈", "유형", "비고"], rows)


def render_claudemd() -> str:
    """Compact table for CLAUDE.md."""
    rows = []
    for s in SOLVERS:
        # Extract last segment for shorter "module" column
        short_mod = s["module"].split(".", 2)[-1].replace(".", "/") + ".py"
        rows.append([f"`{s['name']}`", s["type"], short_mod, s["notes_en"]])
    return render_table(["Name", "Type", "Module", "Notes"], rows)


TARGETS = [
    (REPO / "docs/reference/solver-base.md", render_en_full),
    (REPO / "docs/ko/reference/solver-base.md", render_ko_full),
    (REPO / "CLAUDE.md", render_claudemd),
]


def update_file(path: Path, new_table: str, check: bool) -> bool:
    text = path.read_text()
    pattern = re.compile(
        re.escape(MARK_START) + r".*?" + re.escape(MARK_END),
        re.DOTALL,
    )
    block = f"{MARK_START}\n{new_table}\n{MARK_END}"
    if not pattern.search(text):
        print(f"[skip] {path.relative_to(REPO)}: missing markers")
        return False
    new_text = pattern.sub(block, text)
    if new_text == text:
        return False
    if check:
        print(f"[drift] {path.relative_to(REPO)}")
        return True
    path.write_text(new_text)
    print(f"[updated] {path.relative_to(REPO)}")
    return True


def verify_registered() -> None:
    """Optional: import compass.solvers and verify our SOLVERS matches the registry."""
    try:
        import importlib

        sys.path.insert(0, str(REPO))
        for s in SOLVERS:
            importlib.import_module(s["module"])
        from compass.solvers.base import SolverFactory  # type: ignore

        registered = set(SolverFactory.list_solvers())
    except Exception as e:  # noqa: BLE001
        print(f"[warn] registry verification skipped: {e}")
        return
    listed = {s["name"] for s in SOLVERS}
    missing = registered - listed
    extra = listed - registered
    if missing:
        print(f"[warn] registered but not in SOLVERS: {sorted(missing)}")
    if extra:
        print(f"[warn] in SOLVERS but not registered: {sorted(extra)}")
    if not missing and not extra:
        print(f"[ok] manifest matches registry ({len(listed)} solvers)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if any target would change.")
    args = parser.parse_args()

    drift = False
    for path, render in TARGETS:
        if update_file(path, render(), check=args.check):
            drift = drift or args.check

    verify_registered()

    if args.check and drift:
        print("\nSolver tables out of sync. Run: python scripts/sync_solver_tables.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
