"""Build the report solely from new WSL measurements, including incomplete/failure states."""

from __future__ import annotations

import json
import shutil
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"
REPORT = ROOT / "docs/reports/patterned-cpu-gpu-meep-comparison.md"
PORTABLE_ASSETS = (
    "metrics.json",
    "environment.json",
    "spectra.png",
    "performance.png",
    "convergence.png",
    "timing_jobs.json",
    "linux_requirements_freeze.txt",
    "meep_conda_explicit.txt",
    "fdtdz_conda_explicit.txt",
    "meent_cfs_finished.json",
    "closeout.json",
)


def delta(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def matching_conditions(a, b):
    """Require identical physical/numerical inputs for CPU/GPU parity."""
    return (
        a.get("wavelength_um") == b.get("wavelength_um")
        and a.get("precision") == b.get("precision")
        and a.get("params") == b.get("params")
    )


def epsilon_grid_label(row):
    shape = row.get("metadata", {}).get("epsilon_grid_shape_yx")
    if shape is not None:
        return f"{shape[1]}x{shape[0]} (실제)"
    return f"{row['grid']} (요청)"


def fdtdx_duration(row):
    return row["params"].get("duration", row["params"].get("runtime"))


def meep_time_delta(run):
    """Maximum of both final DFT changes for every measured wavelength."""
    changes = []
    for spectrum in run.get("spectra", []):
        series = spectrum.get("time_series", [])
        if len(series) < 3:
            return None
        changes.extend(delta(a["rta"], b["rta"]) for a, b in pairwise(series[-3:]))
    return max(changes) if len(changes) == 6 else None


def meep_values(run):
    return [s["rta"] for s in sorted(run["spectra"], key=lambda s: s["wavelength_um"])]


def meep_complete(run):
    return (
        run.get("status") == "ok"
        and run.get("method", "time_domain") == "time_domain"
        and {s["wavelength_um"] for s in run.get("spectra", [])} == {0.45, 0.55, 0.65}
    )


def meep_protocol(run, *, include_pml=True):
    protocol = [
        run.get("case"),
        run.get("extinction", 0.02),
        run.get("method", "time_domain"),
        run.get("padding"),
        run.get("fwidth_factor", run["spectra"][0].get("fwidth_factor", 0.15)),
    ]
    if include_pml:
        protocol.append(run.get("pml"))
    return protocol


def fdtdz_physical_validation(native, tmm):
    """Separate successful kernel execution from scoped planar optical accuracy."""
    result = {
        "accepted": False,
        "status": "unverified",
        "tolerance": 0.005,
        "cases": [],
        "comparisons": [],
    }
    if not native or not native.get("spectra"):
        return result
    cases = native["spectra"]
    for index, case in enumerate(cases):
        wavelengths = case.get("wavelength_um", [])
        values = np.asarray(case.get("rta", []))
        if (
            wavelengths != [0.45, 0.55, 0.65]
            or values.shape != (3, 3)
            or not np.isfinite(values).all()
        ):
            result.update(
                status="invalid_result", reason="Missing/nonfinite three-wavelength R/T/A"
            )
            return result
        expected = [
            tmm(
                np.array([1, native.get("slab_index", 1.5), 1]),
                np.array([np.inf, native.get("slab_thickness_um", 0.1), np.inf]),
                wl,
                0.0,
                "TE",
            )
            for wl in wavelengths
        ]
        result["cases"].append({"index": index, "max_tmm_error": delta(values, expected)})
    selected_index = max(
        range(len(cases)),
        key=lambda i: (
            -cases[i]["grid_spacing"],
            cases[i]["duration_um_over_c"],
            cases[i]["transverse_cells"],
        ),
    )
    result.update(
        selected_case=selected_index,
        selected_tmm_error=result["cases"][selected_index]["max_tmm_error"],
        max_tmm_error_all_cases=max(c["max_tmm_error"] for c in result["cases"]),
    )
    for i, a in enumerate(cases):
        for j, b in enumerate(cases[i + 1 :], i + 1):
            same_dx = a["grid_spacing"] == b["grid_spacing"]
            same_width_cells = a["transverse_cells"] == b["transverse_cells"]
            same_time = a["duration_um_over_c"] == b["duration_um_over_c"]
            physical_width_a = a["grid_spacing"] * a["transverse_cells"]
            physical_width_b = b["grid_spacing"] * b["transverse_cells"]
            kind = None
            if same_dx and same_width_cells and not same_time:
                kind = "time"
            elif same_dx and same_time and not same_width_cells:
                kind = "transverse_width"
            elif not same_dx and same_time and abs(physical_width_a - physical_width_b) < 1e-10:
                kind = "grid_and_z_pml"
            if kind:
                result["comparisons"].append(
                    {
                        "kind": kind,
                        "first_case": i,
                        "second_case": j,
                        "max_rta_delta": delta(a["rta"], b["rta"]),
                    }
                )

    def latest_comparison(kind, require_selected=False):
        candidates = [
            c
            for c in result["comparisons"]
            if c["kind"] == kind
            and (not require_selected or selected_index in (c["first_case"], c["second_case"]))
        ]
        return (
            max(
                candidates,
                key=lambda c: (
                    -min(
                        cases[c["first_case"]]["grid_spacing"],
                        cases[c["second_case"]]["grid_spacing"],
                    ),
                    max(
                        cases[c["first_case"]]["duration_um_over_c"],
                        cases[c["second_case"]]["duration_um_over_c"],
                    ),
                ),
            )
            if candidates
            else None
        )

    selected_comparisons = {
        "time": latest_comparison("time", require_selected=True),
        "transverse_width": latest_comparison("transverse_width"),
        "grid_and_z_pml": latest_comparison("grid_and_z_pml", require_selected=True),
    }
    result["selected_comparisons"] = selected_comparisons
    result["input_conditions_verified"] = (
        native.get("case") == "lossless_slab_native"
        and native.get("slab_index") == 1.5
        and native.get("slab_thickness_um") == 0.1
        and native.get("dt_cell_units") == 0.5
        and native.get("precision") == "float32"
        and str(native.get("device", "")).startswith("cuda")
        and all(
            str(c.get("actual_device", "")).startswith("cuda") and c.get("dtype") == "float32"
            for c in cases
        )
    )
    result["timing_conditions_verified"] = native.get("timing_protocol", {}).get(
        "warm_repeats", 0
    ) >= 3 and all(
        len(c.get("warm_s", [])) >= 3
        and c.get("cold_s", 0) > 0
        and all(np.isfinite(t) and t > 0 for t in c["warm_s"])
        and c.get("max_repeat_delta", 1) <= 1e-6
        for c in cases
    )
    result["convergence_verified"] = all(
        c is not None and c["max_rta_delta"] <= 0.005 for c in selected_comparisons.values()
    )
    result["accepted"] = (
        native.get("status") == "ok"
        and result["selected_tmm_error"] <= 0.005
        and result["input_conditions_verified"]
        and result["timing_conditions_verified"]
        and result["convergence_verified"]
    )
    result["status"] = (
        "accepted_scoped_lossless_slab"
        if result["accepted"]
        else "failed_physical_accuracy"
        if result["selected_tmm_error"] > 0.005
        else "unverified_convergence_or_protocol"
    )
    return result


def write_report():
    import sys

    sys.path.insert(0, str(ROOT))
    from compass.solvers.tmm.tmm_core import transfer_matrix_1d

    rows, references = [], []
    for path in sorted(OUT.glob("*.json")):
        if path.name.startswith(("pre_", "metrics")):
            continue
        item = json.loads(path.read_text())
        if not isinstance(item, dict):
            continue
        if (
            "command" in item
            and item.get("status") in ("failed", "timeout", "process_failed")
            and "solver" not in item
        ):
            command = item["command"]
            item["solver"] = (
                command[command.index("--solver") + 1]
                if "--solver" in command
                else "meep_reference"
            )
            item.setdefault("error", "Process failure or timeout; see corresponding log")
        if item.get("solver") == "meent_jax":
            item["device_classification"] = (
                "GPU arrays with CPU eig callback" if item.get("device") == "cuda" else "CPU"
            )
            item["jax_eig_lowering_scope"] = (
                "2x2 capability probe; does not identify the meent eig callback device"
            )
        if "solver" in item:
            item["raw_file"] = path.name
            rows.append(item)
        elif item.get("case") == "pattern" and item.get("spectra"):
            item["raw_file"] = path.name
            references.append(item)
    environment = json.loads((OUT / "environment.json").read_text())
    manifest = {"environment": environment, "runs": rows, "meep": references}
    closeout_path = OUT / "closeout.json"
    closeout = json.loads(closeout_path.read_text()) if closeout_path.exists() else None
    if closeout:
        manifest["closeout"] = closeout
    profiles_path = OUT / "torch_device_profiles.json"
    if profiles_path.exists():
        manifest["torch_device_profiles"] = json.loads(profiles_path.read_text())
    jax_profile_path = OUT / "jax_device_profile.json"
    if jax_profile_path.exists():
        manifest["jax_device_profile"] = json.loads(jax_profile_path.read_text())
    pml_path = OUT / "meep_pml_finished.json"
    if pml_path.exists():
        manifest["meep_pml_validation"] = json.loads(pml_path.read_text())
    cfs_path = OUT / "meent_cfs_finished.json"
    cfs_study = json.loads(cfs_path.read_text()) if cfs_path.exists() else {}
    manifest["meent_cfs_study"] = cfs_study
    slab_path = OUT / "meep_slab_convergence.json"
    if slab_path.exists():
        slab = json.loads(slab_path.read_text())["spectra"]
        for r in slab:
            expected = transfer_matrix_1d(
                np.array([1, 1.5 + 1j * r["extinction"], 1]),
                np.array([np.inf, 0.1, np.inf]),
                r["wavelength_um"],
                0,
                "TE",
            )
            r["tmm_rta"] = list(expected)
            r["max_error"] = delta(r["rta"], expected)
        manifest["meep_slab"] = slab
    else:
        slab = []
    complete = sorted(
        [
            r
            for r in references
            if meep_complete(r) and r["raw_file"].startswith("meep_reference_r")
        ],
        key=lambda r: (r["resolution"], r["duration"]),
    )
    by_file = {r["raw_file"]: r for r in references}
    pml_path = OUT / "meep_pml_finished.json"
    pml_check = json.loads(pml_path.read_text()) if pml_path.exists() else {}
    selected_name = Path(pml_check.get("selected_reference_file") or "").name
    selected = by_file.get(selected_name)
    best = complete[-1] if complete else None
    if selected and meep_complete(selected) and pml_check.get("accepted_reference") is True:
        best = selected
    ref_values = meep_values(best) if best else None
    grid_delta, time_delta, coarse_time_delta = None, None, None
    previous = None
    if best:
        prev = sorted(
            [
                r
                for r in references
                if meep_complete(r)
                and r["resolution"] < best["resolution"]
                and meep_protocol(r) == meep_protocol(best)
            ],
            key=lambda r: (r["resolution"], r["duration"]),
        )
        designated_coarse = by_file.get(Path(pml_check.get("matched_coarser_file") or "").name)
        if designated_coarse in prev and best is selected:
            previous = designated_coarse
        elif prev:
            previous = prev[-1]
        if previous:
            grid_delta = delta(ref_values, meep_values(previous))
            coarse_time_delta = meep_time_delta(previous)
        time_delta = meep_time_delta(best)
    # Recompute width/time evidence from raw runs rather than trusting a summary flag.
    pml_delta, pml_time_delta = None, None
    if best:
        for comparison in pml_check.get("comparisons", []):
            first_name = Path(comparison.get("first_file", "")).name
            second_name = Path(comparison.get("second_file", "")).name
            if best["raw_file"] not in (first_name, second_name):
                continue
            first, second = by_file.get(first_name), by_file.get(second_name)
            if not first or not second or not meep_complete(first) or not meep_complete(second):
                continue
            if (
                first["resolution"] != second["resolution"]
                or meep_protocol(first, include_pml=False)
                != meep_protocol(second, include_pml=False)
                or first.get("pml") == second.get("pml")
            ):
                continue
            change = delta(meep_values(first), meep_values(second))
            times = [meep_time_delta(first), meep_time_delta(second)]
            if (
                all(t is not None for t in times)
                and max(times) <= 0.0005
                and (pml_delta is None or change < pml_delta)
            ):
                pml_delta, pml_time_delta = change, max(times)
    pml_accepted = (
        pml_check.get("status") == "ok"
        and pml_delta is not None
        and pml_delta <= 0.001
        and (
            pml_check.get("accepted_reference") is True
            if "accepted_reference" in pml_check
            else pml_check.get("accepted_base_pml") is True
        )
    )
    accepted = (
        grid_delta is not None
        and grid_delta <= 0.005
        and time_delta is not None
        and time_delta <= 0.0005
        and coarse_time_delta is not None
        and coarse_time_delta <= 0.0005
        and pml_accepted
    )
    manifest["reference_selection"] = {
        "accepted": accepted,
        "selected_file": best["raw_file"] if best else None,
        "matched_coarser_file": previous["raw_file"] if previous else None,
        "grid_delta": grid_delta,
        "time_delta": time_delta,
        "coarse_time_delta": coarse_time_delta,
        "pml_delta": pml_delta,
        "pml_time_delta": pml_time_delta,
    }
    lines = [
        "# WSL 실제 CPU/GPU 및 Meep 기준 전자기 솔버 비교",
        "",
        f"Meep 기준값 수렴 판정: **{'통과' if accepted else '미확정 / 수렴 기준을 아직 충족하지 않음'}**. "
        "이 문서는 `outputs/patterned_cpu_gpu/wsl/`의 새 실행만 사용한다. 이전 Windows 결과는 별도 보존하며 성능 표에 섞지 않는다.",
        "",
        "## 환경과 측정 범위",
        "",
        f"- WSL2 Ubuntu 24.04, Python {environment['python'].split()[0]}, CPU {environment.get('cpu_model', '미확인')} (논리 코어 {environment['cpu_count']}개).",
        f"- GPU: {environment.get('gpu_name', '미확인')}, NVIDIA-SMI 610.57.01 / Windows KMD 610.88, CUDA UMD 13.3. 드라이버 원문은 environment.json에 있다.",
        f"- PyTorch {environment['torch']} (CUDA {environment['torch_cuda']}), JAX {environment['jax']}; 실제 CUDA 행렬 곱 결과를 양쪽에서 확인했다.",
        "- Linux `.venv-wsl`과 Linux 파일시스템의 `/home/edwin/.local/share/compass-miniconda/envs/meep`을 사용한다. Windows `.venv`는 사용하지 않았다. Meep은 공식 conda-forge의 MPI PyMeep 1.35.0이다.",
        "- Meep 기준 계산은 CPU MPI 8개 프로세스를 쓴다. 다른 CPU 솔버의 1개 BLAS/PyTorch 스레드 시간과 코어 수를 같게 맞춘 성능 순위는 아니다. Meep 수렴 실행의 단일 wall time을 warm 3회 중앙값으로 표시하지 않는다.",
        "- NumPy/BLAS/OMP/MKL 및 PyTorch 스레드 1개. JAX/XLA 내부 전체 스레드 수를 1개로 제한했다는 뜻은 아니다. 복소 RCWA 배정밀도, FDTDX 실수 필드 배정밀도 + 복소 DFT 배정밀도.",
        "- 프로세스 시작, 공통 NumPy/PyTorch import, 장치 초기화와 소형 HLO probe는 시간에서 제외된다. cold는 완전히 새로운 OS 프로세스의 총 시작 시간이 아닌 어댑터 최초 run() 시간이다.",
        "- 시간은 geometry/source 설정을 제외한 `run()`과 NumPy 결과 전송까지다. 최초 실행은 지연 import/JAX 컴파일을 포함하고, warm은 3회 중앙값이다. PyTorch CUDA는 전후 동기화하며 JAX는 결과의 호스트 변환으로 완료를 보장한다.",
        "- PyTorch 메모리는 max_memory_allocated/reserved, JAX는 device.memory_stats의 peak_bytes_in_use다. 드라이버 메모리와 전체 프로세스 VRAM의 같은 지표가 아니다.",
        "- 수렴용 단일 실행과 반복 성능 측정을 구분한다. Meep 계산과 부하가 겹친 초기 시간은 `pre_` 자료로 분리하거나 재측정한다.",
        "",
        "[공식 Meep 설치](https://meep.readthedocs.io/en/master/Installation/)의 Conda 설치법과 [복소 유전율/전도도 정의](https://meep.readthedocs.io/en/master/Materials/)를 사용했다. PyPI의 다른 `meep` 패키지는 설치하지 않았다.",
        "",
        "## 구조와 물리량",
        "",
        "주기 0.8x0.8 µm, 공기 입출사, 수직 TE(Ey), 450/550/650 nm. 아래에서부터 n=1.45 spacer 0.12 µm, 높이 0.24 µm의 n=2.4+0.02i 기둥 2개, n=1.6 cap 0.08 µm이다. 기둥 (중심 x,y; 크기 x,y)은 (-0.2,-0.15;0.25,0.2), (0.175,0.175;0.2,0.25) µm이다.",
        "R/T는 모든 전파 회절차수의 총 파워다. A=1-R-T는 잔차이며 합계 1을 정확도 증거로 사용하지 않는다. 이번 경로에서 clipping을 제거했다. 합성 구조에는 photodiode가 없으며 QE 비교는 하지 않는다.",
        "torcwa/meent/fmmax는 사각형 기저 (2N+1)², grcwa는 원형 기저다. grcwa의 실제 모드 수를 기록하며 N 숫자만으로 동일 기저라고 간주하지 않는다.",
        "격자 표의 요청값은 epsilon_grid 최소 크기다. 새 metadata에 실제 epsilon_grid_shape_yx가 있으면 주요 결과 표에 실제 x/y 크기를 표시한다. 독립 차수·격자 수렴의 그룹 기준은 요청값이며 이번 benchmark는 grid_multiplier=0을 사용한다.",
        "",
        "## Meep 박막 검증 (TMM 해석해)",
        "",
        "공기/n=1.5+ik/공기, 두께 0.1 µm, k=0/0.005/0.05, 세 파장. 정상 입사 균일 박막은 1D Meep으로 독립 검증하고 COMPASS 3D 어댑터는 실제 통합 테스트로 검증한다. 각 재료는 목표 주파수에서 ε=(n+ik)²와 일치하도록 D_conductivity=2πf Im(ε)/Re(ε)를 사용한다.",
        "",
        "| 해상도 (points/µm) | 9개 조건의 최대 R/T/A 오차 | 시간 40→80 최대 변화 |",
        "|---:|---:|---:|",
    ]
    for resolution in sorted({r["resolution"] for r in slab}):
        group = [r for r in slab if r["resolution"] == resolution]
        td = max(delta(r["time_series"][-1]["rta"], r["time_series"][-2]["rta"]) for r in group)
        lines.append(f"| {resolution} | {max(r['max_error'] for r in group):.6g} | {td:.3e} |")
    lines += [
        "",
        "## 실제 GPU 및 FDTD 박막 검증",
        "",
        "| 솔버 / 장치 | k | 격자 간격 (µm) | 파장 수 | 최대 TMM R/T/A 오차 | 구현 상태 | 원시 파일 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        if row.get("case") == "slab" and row.get("status") == "ok" and row.get("rta"):
            expected = [
                transfer_matrix_1d(
                    np.array([1, 1.5 + 1j * row["extinction"], 1]),
                    np.array([np.inf, 0.1, np.inf]),
                    wl,
                    0.0,
                    "TE",
                )
                for wl in row["wavelength_um"]
            ]
            error = delta(row["rta"], expected)
            grid_label = (
                row.get("params", {}).get("grid_spacing")
                if row["solver"].startswith("fdtd")
                else f"N={row.get('order')}, eps={epsilon_grid_label(row)}"
            )
            historical = row["raw_file"] in {
                "fdtdx_cuda_slab_native.json",
                "fdtdx_cuda_slab_absorbing_dx0125.json",
            }
            implementation = "이전 탐색 구현; 현행 검증에서 제외" if historical else "현행 측정"
            lines.append(
                f"| {row['solver']} / {row['device']} | {row['extinction']} | {grid_label} | {len(row['wavelength_um'])} | {error:.6g} | {implementation} | `{row['raw_file']}` |"
            )
    lines += [
        "",
        "## Meep 패턴 격자·시간 수렴",
        "",
        "PML 밖 공기에 소스/면 모니터를 놓고 동일 셀의 빈 공간 계산에서 얻은 DFT 필드를 반사 모니터에서 뺀다. 입사 플럭스는 -z 방향의 음수, 반사 플럭스는 양수다. 전도도는 파장마다 다시 계산한다. 전체 셀 면적의 DFT 플럭스를 적분하므로 회절차수를 빠뜨리지 않는다.",
        "펄스 폭과 air/PML 크기를 바꾼 탐색 계산은 원시 자료에 보존한다. 최종 격자 비교는 동일한 좁은 펄스와 air/PML 설정만 사용하며 PML 폭의 독립 수렴 시험으로 대체하지 않는다. 광펄스 종료 후 시간의 누적 DFT 결과를 저장하며, 격자 판정은 인접 최고 해상도 변화 ≤0.005, 시간 판정은 마지막 두 시간 변화 각각 ≤0.0005이다. PML 판정은 동일한 최고 격자·소스·시간 설정에서 인접 PML 폭 변화 ≤0.001이다. 기본 0.35→0.7 µm가 실패하면 더 넓은 인접 PML 쌍과 선택한 PML의 낮은 해상도를 추가 검증한 뒤 기준을 변경한다. 이전 PML의 결과를 새 설정의 수렴 증거로 대신하지 않는다. 이 변화 기준은 엄밀한 오차 상계가 아니다.",
        "",
        "| 해상도 | post-source 시간 상한 | air / PML (µm) | 계산된 파장 수 | 상태 |",
        "|---:|---:|---:|---:|---|",
    ]
    for r in sorted(references, key=lambda r: (r["resolution"], r["duration"])):
        lines.append(
            f"| {r['resolution']} | {r['duration']} | {r.get('padding')} / {r.get('pml')} | {len(r['spectra'])} | {r['status']} |"
        )
    lines += [
        "",
        "주 기준 실행의 파장별 실제 종료시간과 마지막 두 누적 DFT 변화다. 시간 상한에 도달한 것과 시간 수렴 통과를 구분한다. 아직 계산하지 않은 파장을 0 또는 수렴한 값으로 채우지 않는다.",
        "",
        "| 원시 파일 / 해상도 | λ (nm) | 실제 post-source 종료 (µm/c) | 이전 구간 / ΔRTA | 마지막 구간 / ΔRTA | 시간 판정 |",
        "|---|---:|---:|---|---|---|",
    ]
    primary = sorted(
        [
            r
            for r in references
            if (r["raw_file"].startswith("meep_reference_r") or r is best or r is previous)
            and r.get("method", "time_domain") == "time_domain"
        ],
        key=lambda r: (r["resolution"], r["duration"]),
    )
    for run in primary:
        spectra = {s["wavelength_um"]: s for s in run["spectra"]}
        for wavelength in (0.45, 0.55, 0.65):
            series = spectra.get(wavelength, {}).get("time_series", [])
            changes = []
            for a, b in pairwise(series[-3:]):
                change = delta(a["rta"], b["rta"])
                changes.append((f"{a['duration']}→{b['duration']} / {change:.6g}", change))
            stopped = series[-1]["duration"] if series else "미측정"
            previous_label = changes[-2][0] if len(changes) >= 2 else "미측정"
            final = changes[-1][0] if changes else "미측정"
            passed = len(changes) == 2 and max(v for _, v in changes) <= 0.0005
            judgment = "통과" if passed else ("미충족" if series else "미측정 / 진행 중")
            lines.append(
                f"| `{run['raw_file']}` / {run['resolution']} | {wavelength * 1000:.0f} | {stopped} | {previous_label} | {final} | {judgment} |"
            )
    lines += [
        "",
        f"비교에 선택한 Meep 파일: `{best['raw_file'] if best else '미측정'}`; PML {best.get('pml') if best else '미측정'} µm. 짝을 이룬 낮은 해상도 파일: `{previous['raw_file'] if previous else '미측정'}`.",
        f"동일 PML의 인접 최고 격자 변화: {grid_delta}; 최고/이전 격자의 마지막 두 구간 최대 시간 변화: {time_delta} / {coarse_time_delta}. 선택한 PML 변화: {pml_delta}; 해당 두 계산의 최대 시간 변화: {pml_time_delta}. PML 판정: {pml_check.get('status', '미측정')} / 선택 PML 허용 {pml_accepted}. 수렴 판정: {'통과' if accepted else '미확정'}.",
        "",
        "### Meep PML 독립 검증",
        "",
        f"원래 PML 허용: {pml_check.get('accepted_base_pml', False)}. 원시 자료에서 재계산한 선택 PML 검증의 마지막 두 구간 최대 시간 변화: {pml_time_delta}. 전체 기준값 승인은 두 격자와 PML 비교 양쪽 계산의 시간 변화가 모두 ≤0.0005일 때만 가능하다.",
        "",
        "| PML 폭 변화 (µm) | 최대 ΔRTA |",
        "|---:|---:|",
    ]
    for comparison in pml_check.get("comparisons", []):
        change = comparison.get("max_rta_delta")
        change_label = f"{change:.6g}" if change is not None else "미측정 / 실패"
        lines.append(f"| {comparison['first_pml']}→{comparison['second_pml']} | {change_label} |")
    if not pml_check.get("comparisons"):
        lines.append("| 미측정 / 진행 중 | — |")
    lines += [
        "",
        "## CPU/GPU 동일 조건의 성능과 결과 차이",
        "",
        "| 솔버 | N / 격자 | 실제 모드 | CPU cold / warm (s) | GPU cold / warm (s) | 가속률 | CPU↔GPU 최대 ΔRTA | Meep 최대 ΔRTA | GPU allocator peak (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    success = [
        r
        for r in rows
        if r.get("status") == "ok"
        and r.get("case") == "pattern"
        and len(r.get("wavelength_um", [])) == 3
    ]
    paired = []
    for a in success:
        if (
            a.get("device") != "cpu"
            or a.get("warm_median_s") is None
            or a.get("timing_context") != "isolated"
        ):
            continue
        matches = [
            b
            for b in success
            if b.get("device") == "cuda"
            and b.get("timing_context") == "isolated"
            and b["solver"] == a["solver"]
            and matching_conditions(a, b)
            and b.get("warm_median_s") is not None
        ]
        b = matches[-1] if matches else None
        err = delta(a["rta"], ref_values) if ref_values else None
        gpu = f"{b['cold_s']:.3f} / {b['warm_median_s']:.3f}" if b else "해당 없음 / 미측정"
        speed = f"{a['warm_median_s'] / b['warm_median_s']:.3f}x" if b else "—"
        change = f"{delta(a['rta'], b['rta']):.3e}" if b else "—"
        memory = (
            b.get(
                "torch_peak_allocated_bytes", b.get("jax_memory_stats", {}).get("peak_bytes_in_use")
            )
            if b
            else None
        )
        memory = f"{memory / 2**20:.1f}" if memory is not None else "—"
        lines.append(
            f"| {a['solver']} | {a['order']} / {epsilon_grid_label(a)} | {a.get('actual_modes')} | {a['cold_s']:.3f} / {a['warm_median_s']:.3f} | {gpu} | {speed} | {change} | {err if err is not None else '미확정'} | {memory} |"
        )
        if b:
            paired.append((a, b))
    lines += [
        "",
        "meent/JAX의 GPU 열은 GPU 배열 연산 + CPU 고유값 풀이의 혼합 경로다. 완전 GPU 실행의 가속률로 해석하지 않는다.",
        "위 Meep 오차는 최고 측정 격자와의 차이다. Meep 기준값이 미확정이면 정확도 순위로 사용하지 않는다. 낮은 차수의 빠른 결과를 정확한 솔버로 추천하지 않는다.",
        "",
        "## 고차수 CPU/GPU 결과 일치",
        "",
        "반복 시간 측정과 별개로 동일 입력의 고차수 결과를 비교한다. 단일 수렴 실행에는 warm 가속률을 붙이지 않는다.",
        "",
        "| 솔버 | N / 유전율 격자 | 실제 모드 | CPU↔GPU 최대 ΔRTA |",
        "|---|---:|---:|---:|",
    ]
    parity_count = 0
    for a in success:
        if a.get("device") != "cpu" or a.get("order", 0) < 11:
            continue
        matches = [
            b
            for b in success
            if b.get("device") == "cuda"
            and b["solver"] == a["solver"]
            and matching_conditions(a, b)
        ]
        if matches:
            b = matches[-1]
            lines.append(
                f"| {a['solver']} | {a['order']} / {epsilon_grid_label(a)} | {a.get('actual_modes')} | {delta(a['rta'], b['rta']):.3e} |"
            )
            parity_count += 1
    if not parity_count:
        lines.append("| 미측정 / 진행 중 | — | — | — |")
    lines += [
        "",
        "## 차수와 유전율 격자의 독립 수렴",
        "",
        "| 솔버 / 장치 | 격자 | N 변화 | 모드 수 변화 | 최대 ΔRTA |",
        "|---|---:|---:|---:|---:|",
    ]
    rcwa = [r for r in success if not r["solver"].startswith("fdtd")]
    for name, device, grid in sorted({(r["solver"], r["device"], r["grid"]) for r in rcwa}):
        group = {
            r["order"]: r
            for r in rcwa
            if r["solver"] == name and r["device"] == device and r["grid"] == grid
        }
        for na, nb in pairwise(sorted(group)):
            a, b = group[na], group[nb]
            lines.append(
                f"| {name} / {device} | {grid} | {na}→{nb} | {a.get('actual_modes')}→{b.get('actual_modes')} | {delta(a['rta'], b['rta']):.6g} |"
            )
    lines += ["", "| 솔버 / 장치 | N | 격자 변화 | 최대 ΔRTA |", "|---|---:|---:|---:|"]
    for name, device, order in sorted({(r["solver"], r["device"], r["order"]) for r in rcwa}):
        group = {
            r["grid"]: r
            for r in rcwa
            if r["solver"] == name and r["device"] == device and r["order"] == order
        }
        for ga, gb in pairwise(sorted(group)):
            a, b = group[ga], group[gb]
            lines.append(
                f"| {name} / {device} | {order} | {ga}→{gb} | {delta(a['rta'], b['rta']):.6g} |"
            )
    lines += [
        "",
        "### meent CFS 별도 알고리즘 검증",
        "",
        "`meent_torch_cfs`는 Torch backend의 fourier_type=1 (continuous Fourier series)이다. 기존 meent/torch의 fourier_type=0, enhanced_dfs=True (enhanced discrete Fourier series)와 별도 조건으로 기록한다. 기본 설정과 기존 측정은 유지했다. CFS는 작은 입력 격자에서 해석적으로 계수를 적분하므로 같은 격자의 DFS FFT Nyquist 제한을 그대로 적용하지 않는다.",
        "`epsilon_grid_shape_yx`는 어댑터가 전달한 실제 입력 격자다. DFS의 `fourier_work_grid_shape_yx`는 설치된 upstream 반복 규칙으로 유도한 FFT 작업 격자이며 별도의 runtime 배열 관측치나 GPU 메모리 측정값이 아니다. enhanced DFS는 N>0일 때 축별 입력을 (4N+2)배 반복한다. 예를 들어 N9/input128은 4864x4864, N29/input256은 30208x30208이며 후자의 complex128 배열 하나만 약 13.6 GiB다. 단일 배열 추정치와 allocator peak를 동일하게 해석하지 않는다. CFS는 이 FFT 작업 격자를 만들지 않으므로 해당 metadata는 null이다.",
        "CFS의 정확도 계산은 repeats=0으로 수집하며 기존 단독 N9 DFS의 warm 가속률을 CFS 또는 고차수로 외삽하지 않는다. GPU 잠금으로 직렬화하지만 CPU 정확도 작업과 겹칠 수 있다. 550 nm CPU 박막 API smoke는 패턴의 세 파장 검증과 별도 조건이다.",
        f"추가 CFS workflow 상태: **{cfs_study.get('status', '미측정 / 대기')}**; 박막·장치 gate: {cfs_study.get('low_order_validation_passed', '미측정')}; 차수 수렴: {cfs_study.get('order_converged', '미측정')}; 입력 격자 수렴: {cfs_study.get('grid_converged', '미측정')}. 이 상태만으로 Meep 대비 정확도 통과를 주장하지 않는다.",
        "DFS↔CFS 차이 0.005는 두 알고리즘의 동일 차수 결과를 진단하는 기준이다. 이를 초과하면 같은 정확도로 간주하지 않으며 CFS 자체의 차수·격자 수렴과 최종 Meep 오차를 따로 평가한다. 해석 박막 및 CPU/GPU 일치 검증이 실패하면 고차수 CFS 확장을 중단한다.",
        "",
        "| CFS 검증 | 최대 ΔRTA | 기준 | 판정 | 근거 파일 |",
        "|---|---:|---:|---|---|",
    ]
    for check in cfs_study.get("checks", []):
        value = check.get("max_rta_delta")
        label = f"{value:.6g}" if value is not None else "미측정 / 실패"
        evidence = [check[k] for k in ("file", "first_file", "second_file") if check.get(k)]
        lines.append(
            f"| {check['kind']} | {label} | {check['tolerance']} | {check['passed']} | "
            + ", ".join(f"`{f}`" for f in evidence)
            + " |"
        )
    if not cfs_study.get("checks"):
        lines.append("| 대기 / 아직 측정되지 않음 | — | — | 미확정 | — |")
    cfs_smoke = next(
        (r for r in rows if r["raw_file"] == "meent_torch_cfs_cpu_slab_api_smoke.json"), None
    )
    if cfs_smoke and cfs_smoke.get("status") == "ok":
        lines += [
            "",
            f"별도 CPU CFS 550 nm 흡수 박막 smoke의 최대 TMM 오차: {cfs_smoke.get('max_error_vs_tmm')}; `meent_torch_cfs_cpu_slab_api_smoke.json`. GPU 패턴 검증을 대신하지 않는다.",
        ]
    lines += [
        "",
        "- [CFS workflow 원시 판정](../../outputs/patterned_cpu_gpu/wsl/meent_cfs_finished.json)",
        "",
    ]
    lines += [
        "",
        "### FDTDX 유전율 격자 수렴",
        "",
        "동일한 실행시간끼리만 공간 격자를 비교한다. PML 셀 수는 물리적 폭을 일정하게 유지하도록 격자 간격에 반비례한다.",
        "",
        "| 실행시간 (µm/c) | GPU 격자 간격 변화 (µm) | 최대 R/T/A 변화 | 최고 Meep 격자 대비 차이 |",
        "|---:|---:|---:|---:|",
    ]
    fdtdx_rows = [r for r in success if r["solver"] == "fdtdx" and r["device"] == "cuda"]
    for duration in sorted({fdtdx_duration(r) for r in fdtdx_rows}):
        finer = {
            r["params"]["grid_spacing"]: r for r in fdtdx_rows if fdtdx_duration(r) == duration
        }
        for a, b in pairwise(sorted(finer, reverse=True)):
            error = delta(finer[b]["rta"], ref_values) if ref_values else "미확정"
            lines.append(
                f"| {duration} | {a}→{b} | {delta(finer[a]['rta'], finer[b]['rta']):.6g} | {error} |"
            )
    lines += [
        "",
        "### FDTDX 실행시간 수렴",
        "",
        "동일한 공간 격자와 PML 설정을 유지하고 총 실행시간을 늘린 독립 계산이다. Meep의 post-source 시간과 직접 같은 시간 정의로 간주하지 않는다.",
        "",
        "| GPU 격자 간격 (µm) | 실행시간 변화 (µm/c) | 최대 ΔRTA |",
        "|---:|---:|---:|",
    ]
    for spacing in sorted({r["params"]["grid_spacing"] for r in fdtdx_rows}, reverse=True):
        group = {fdtdx_duration(r): r for r in fdtdx_rows if r["params"]["grid_spacing"] == spacing}
        for ta, tb in pairwise(sorted(group)):
            a, b = group[ta], group[tb]
            if a["params"].get("pml_layers") == b["params"].get("pml_layers"):
                lines.append(f"| {spacing} | {ta}→{tb} | {delta(a['rta'], b['rta']):.6g} |")
    lines += [
        "",
        "## 실측 R/T/A (최고 측정 차수; 수렴 판정과 별개)",
        "",
        "| 솔버 | N / 격자 | λ (nm) | R | T | A |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if best:
        for r in best["spectra"]:
            lines.append(
                f"| Meep | res={best['resolution']} | {r['wavelength_um'] * 1000:.0f} | "
                + " | ".join(f"{v:.8f}" for v in r["rta"])
                + " |"
            )
    highest = []
    for name in sorted({r["solver"] for r in success}):
        if name == "fdtdx":
            candidates = [r for r in success if r["solver"] == name]
            finest = min(r["params"]["grid_spacing"] for r in candidates)
            candidates = [r for r in candidates if r["params"]["grid_spacing"] == finest]
        else:
            candidates = [r for r in success if r["solver"] == name]
        if not candidates:
            candidates = [r for r in success if r["solver"] == name]
        row = max(
            candidates,
            key=lambda r: (
                fdtdx_duration(r) if name == "fdtdx" else r["order"],
                r["grid"],
                r["device"] == "cpu",
            ),
        )
        highest.append(row)
        setting = (
            f"dx={row['params']['grid_spacing']}, t={fdtdx_duration(row)}"
            if name == "fdtdx"
            else f"{row['order']} / {epsilon_grid_label(row)}"
        )
        for wl, rta in zip(row["wavelength_um"], row["rta"]):
            lines.append(
                f"| {name} / {row['device']} | {setting} | {wl * 1000:.0f} | "
                + " | ".join(f"{v:.8f}" for v in rta)
                + " |"
            )
    lines += [
        "",
        "최고 측정 설정의 Meep 차이 (수렴 판정과 별개):",
        "",
        "| 솔버 / 장치 | 최대 ΔRTA (세 파장 전체) | 원시 파일 |",
        "|---|---:|---|",
    ]
    for row in highest:
        error = f"{delta(row['rta'], ref_values):.6g}" if ref_values else "미확정"
        lines.append(f"| {row['solver']} / {row['device']} | {error} | `{row['raw_file']}` |")
    native = next((r for r in rows if r.get("raw_file") == "fdtdz_native.json"), None)
    install = next((r for r in rows if r.get("solver") == "fdtdz_install"), None)
    native_path, install_path = OUT / "fdtdz_native.json", OUT / "fdtdz_environment.json"
    native_historical = (
        native is not None
        and install is not None
        and install.get("status") == "ok"
        and native_path.exists()
        and install_path.exists()
        and native_path.stat().st_mtime < install_path.stat().st_mtime
    )
    native_physics = fdtdz_physical_validation(native, transfer_matrix_1d)
    if native_historical:
        native_physics.update(
            accepted=False, status="historical_environment_pending_new_measurement"
        )
    manifest["fdtdz_physical_validation"] = native_physics
    native_status = (
        "이전 환경의 이력; 현 설치 환경의 새 수치 측정 대기"
        if native_historical
        else native.get("status")
        if native
        else "미측정"
    )
    lines += [
        "",
        "## fdtdz 별도 무손실 native 검증",
        "",
        "재료 손실과 주기적 x/y 경계 API가 없는 CUDA 커널이다. 원래 패턴과 혼합하지 않고 n=1.5, 두께 0.1 µm 무손실 박막의 정상 입사 조건을 사용한다. 넓은 균일 소스의 중심에서 측면 경계 신호가 도달하기 전의 펄스를 DFT하고, 공기 중 reference-subtracted Ey 진폭으로 R/T를 구한다. 횡방향 폭과 시간 창을 변화시켜 1D 평면파 조건을 확인한다. z PML은 8셀 고정이므로 dx 감소 시 물리적 PML 폭도 0.2→0.1 µm로 바뀐다. 마지막 두 조건은 격자와 PML이 함께 변한 검증이며 독립적인 공간 격자 수렴으로 부르지 않는다. 이 추출기는 패턴의 모든 회절 파워를 구하는 용도가 아니다.",
        f"설치 상태: {install.get('status') if install else '진행 중'}; 수치 실행 상태: {native_status}.",
        "JAX 0.4.34/CUDA 12.6 전용 환경을 사용하며 32-bit, z 길이 제한 및 수동 launch_params를 명시했다. CPU lowering 지원 여부도 별도 시도한다. 원래 2D 주기 패턴의 CPU/GPU 가속률 표에 넣지 않는다.",
        "fdtdz_environment.json의 설치 성공 시각보다 오래된 native 결과는 이전 환경의 이력으로만 남기며 현 환경의 수치 실행 판정으로 쓰지 않는다. 아래 native 시간은 GPU 잠금으로 직렬화한 정확도 단계의 측정이며 CPU 정확도 계산과는 겹칠 수 있다. 단독 부하의 N9 성능 표와 같은 격리 조건으로 간주하지 않는다.",
        "",
        f"물리 정확도 판정: **{native_physics['status']}** (승인 {native_physics['accepted']}). 커널 실행 status=ok는 정확도 승인이 아니다. 원래 손실성 주기 패턴의 속도 순위에서는 native 결과를 항상 제외한다. 물리 검증을 통과하지 않은 기준선은 별도 박막의 속도 추천에서도 제외한다.",
        f"모든 측정 조건의 최대 TMM 오차: {native_physics.get('max_tmm_error_all_cases', '미측정')}; 최종 선택 조건의 TMM 오차: {native_physics.get('selected_tmm_error', '미측정')}. 선택 조건 TMM 오차와 아래 수렴 변화가 각각 ≤0.005이고 입력·장치·반복 조건도 확인될 때만 별도 무손실 박막 범위에서 승인한다. 실패한 raw 값과 음수 A도 숨기거나 자르지 않는다.",
        "",
        "| dx (µm) | x/y 셀 | 시간 창 (µm/c) | native cold / warm (s) | 최대 TMM ΔRTA | 이전 조건 대비 최대 변화 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    previous_native = None
    for case in native.get("spectra", []) if native and not native_historical else []:
        expected = [
            transfer_matrix_1d(
                np.array([1, 1.5, 1]), np.array([np.inf, 0.1, np.inf]), wl, 0.0, "TE"
            )
            for wl in case["wavelength_um"]
        ]
        change = delta(case["rta"], previous_native["rta"]) if previous_native else None
        lines.append(
            f"| {case['grid_spacing']} | {case['transverse_cells']} | {case['duration_um_over_c']} | {case['cold_s']:.4f} / {case['warm_median_s']:.4f} | {delta(case['rta'], expected):.6g} | {change} |"
        )
        previous_native = case
    lines += [
        "",
        "독립 변화 비교는 같은 dx/횡폭의 시간 창, 같은 dx/시간의 횡폭, 같은 물리적 횡폭/시간의 격자+zPML 쌍을 구분한다. 최종 조건의 시간 쌍이 없으면 시간 수렴을 승인하지 않는다. 횡폭 검증이 낮은 해상도에만 있으면 그 범위를 아래 case 번호로 드러낸다. 격자+zPML 비교는 순수 격자 수렴으로 부르지 않는다.",
        "",
        "| 변화 종류 | case 번호 (0부터) | 최대 ΔRTA |",
        "|---|---|---:|",
    ]
    for comparison in native_physics.get("comparisons", []):
        lines.append(
            f"| {comparison['kind']} | {comparison['first_case']}→{comparison['second_case']} | {comparison['max_rta_delta']:.6g} |"
        )
    lines += [
        "",
        f"입력/실제 GPU 조건 확인: {native_physics.get('input_conditions_verified', False)}; 반복 시간 조건 확인: {native_physics.get('timing_conditions_verified', False)}; 수렴 변화 확인: {native_physics.get('convergence_verified', False)}.",
    ]
    if native and not native_historical:
        lines += [
            "",
            f"실제 장치: {native.get('device')}; CPU 실행 판정: {native.get('cpu_status', '미측정')}. 이 커널의 별도 workspace는 JAX allocator 통계에 포함되지 않아 프로세스 VRAM peak로 보고하지 않는다.",
        ]
    lines += [
        "",
        "## 실제 장치, 실패와 제한",
        "",
        "- torcwa/meent/torch: 실제 torch.linalg.eig 입력 장치를 기록한다. torcwa upstream은 autograd용 고유벡터를 CPU에도 보관하므로 GPU→CPU 전송이 완전히 없는 구현은 아니다.",
        "- fmmax: JAX 장치를 명시하고 실제 eig 호출과 HLO lowering을 기록한다. 설치된 JAX의 `cusolver_geev_ffi`는 GPU 고유값 계산이다. 옛 JAX에서의 host LAPACK 경로와 혼동하지 않는다.",
        "- meent/JAX: 요청 장치를 전달해 GPU 배열 계산은 수행하지만 upstream `primitives.eig`가 CPU에 JIT한 함수를 pure_callback으로 호출한다. CPU/GPU 혼합 계산이며 완전 GPU 솔버로 세지 않는다.",
        "- grcwa 및 meent/numpy: CPU 전용. CUDA 요청을 지원하는 것처럼 보고하지 않는다.",
        "- flaport: damping과 유전율 텐서를 필드 장치에 배치하고 전체 필드의 매 스텝 CPU 복사를 제거했다. 하지만 하드 소스/플럭스 정규화의 정확도 문제는 남아 있다. 실제 무손실 박막의 비물리 결과 때문에 정확한 패턴 속도 비교에서는 제외한다.",
        "- fdtdz 1.1.3: 최초 빌드는 nvcc 부재로 실패했으나 별도 CUDA 12.6 컴파일러/JAX 0.4.34 환경에서 재설치했다. 설치와 수치 실행은 구분하며 아래 native 검증 결과를 따로 기록한다. 기존 COMPASS simulate API 및 주기적 x/y 경계/재료 손실과 호환되지 않으므로 원래 손실성 패턴의 정상 어댑터로 세지 않는다.",
        "- FDTDX 0.6.3: 기존 simulate API 실패를 실제 재현한 뒤 place_objects/run_fdtd와 PhasorPoyntingFluxDetector로 구현했다. 손실은 sigma=omega*eps0*eps''에 내부 격자 길이 단위 변환을 적용한다. 기준 빈 셀의 복소 DFT 필드를 빼고 플럭스를 정규화한다. 패턴 검증과 격자/시간 수렴 결과에 따라 사용 범위를 판정한다. QE는 미지원으로 NaN을 반환한다.",
        "",
        "각 상태는 원시 JSON의 status다. 미완료·실패에는 대체 수치를 넣지 않고 정상 성능/정확도 표에서 제외한다. 이후 수정으로 해결된 이전 실패도 재현 기록으로 남긴다.",
        "",
        "| 실제 실패/진단 파일 | 원시 상태 | 사유 |",
        "|---|---|---|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            error = (
                r.get("error", r.get("reason", "로그 확인"))
                .splitlines()[-1]
                .replace("|", "/")[:240]
            )
            state = r["status"]
            if r["raw_file"] == "fdtdz_native.json" and native_historical:
                state += " (이전 환경 이력; 현 환경 수치 검증 대기)"
            lines.append(f"| {r['raw_file']} | {state} | {error} |")
    lines += [
        "",
        "## 구현·검증과 재현",
        "",
        "기존 비대칭 패턴의 torcwa/grcwa 입력 전치는 유지했다. torcwa field_xy의 [x,y] 출력을 QE의 [y,x] 격자로 돌려놓고 photodiode가 없으면 QE 재구성을 생략한다. meent에는 Torch/JAX 장치를 전달한다. RCWA의 유전율 격자는 epsilon_grid로 독립 제어하며 grid_multiplier=0이면 자동 oversampling을 끈다. meent/fmmax도 이 설정을 따르도록 수정했고 기본 grid_multiplier=3은 유지한다. 새 metadata의 epsilon_grid_requested는 요청 최소 크기, epsilon_grid_shape_yx는 실제 [y,x] 샘플 수다. 수정 직전 완료된 WSL meent/fmmax 결과 18개는 기존 max(epsilon_grid,3*(2N+1)) 식으로 확인했으며 모두 요청 격자와 같았다. 이 검토는 과거 runtime metadata를 새로 측정한 것처럼 채운 것이 아니다. 잔차 흡수를 자르지 않는다.",
        "Meep은 air padding, 고정 post-source 종료시간, 약한 손실 보존, 금속의 단일 주파수 Drude 매핑, DFT 필드 수명과 입사 파워 기준 흡수 적분을 수정했다. 전체 영역 박막 회귀 검증을 포함하지만 비대칭 개별 PD의 QE 동등성 검증은 이번 R/T/A 비교 범위 밖이다.",
        "",
        "```bash",
        ".venv-wsl/bin/python scripts/benchmark_wsl_solvers.py --environment",
        ".venv-wsl/bin/python scripts/benchmark_wsl_solvers.py --jobs docs/public/reports/patterned-cpu-gpu-meep-wsl/timing_jobs.json",
        "```",
        "",
        "위 반복 성능 측정은 다른 계산이 없는 상태에서 먼저 끝낸다. 아래는 정확도·수렴 workflow다. 각 driver를 하나씩만 실행하고 기존 실행이 끝나기 전에 중복 시작하지 않는다. 완료 marker를 기다리는 driver와 GPU 파일 잠금이 의존 순서 및 GPU 동시 실행을 조절한다. 이 명령은 재현 절차이며 명령을 기재한 사실을 완료 증거로 사용하지 않는다. 각 실제 완료·실패는 위 표와 원시 JSON에 따른다.",
        "",
        "전용 fdtdz 환경의 설치·검증 marker도 필요하므로 Linux conda 환경을 위 경로에 준비한 뒤 installer를 먼저 실행한다. 의존 버전은 첨부한 pip/conda 고정 목록에 기록했다.",
        "",
        "```bash",
        ".venv-wsl/bin/python scripts/install_fdtdz_legacy.py > outputs/patterned_cpu_gpu/wsl/fdtdz_installer_retry.log 2>&1",
        "mkdir -p .cache",
        "touch outputs/patterned_cpu_gpu/wsl/finish.log",
        ".venv-wsl/bin/python scripts/finish_wsl_benchmarks.py > outputs/patterned_cpu_gpu/wsl/finish.log 2>&1 &",
        ".venv-wsl/bin/python scripts/check_meep_pml_convergence.py > outputs/patterned_cpu_gpu/wsl/meep_pml.log 2>&1 &",
        ".venv-wsl/bin/python scripts/run_fdtdz_validation.py > outputs/patterned_cpu_gpu/wsl/fdtdz_validation_driver_retry.log 2>&1 &",
        ".venv-wsl/bin/python scripts/benchmark_high_order_gpu.py > outputs/patterned_cpu_gpu/wsl/high_gpu.log 2>&1 &",
        ".venv-wsl/bin/python scripts/check_fdtdx_time_convergence.py > outputs/patterned_cpu_gpu/wsl/fdtdx_time.log 2>&1 &",
        ".venv-wsl/bin/python scripts/benchmark_meent_cfs.py > outputs/patterned_cpu_gpu/wsl/meent_cfs.log 2>&1 &",
        "wait",
        ".venv-wsl/bin/python scripts/report_wsl_solvers.py",
        "```",
        "",
        "Meep 재현: `python scripts/finish_wsl_benchmarks.py`가 conda 환경의 mpirun -np 8 python scripts/benchmark_meep_reference.py를 호출한다. 고해상도에서는 좁은 광펄스와 비등간격 시간 체크포인트를 사용하고 마지막 두 구간의 R/T/A 변화가 각각 0.0005 이하일 때 조기 종료한다. 원시 JSON에 해상도, padding/PML, 펄스 폭, 실제 시간, signed flux, MPI 수를 기록했다. 수렴용 JSON에는 warm 반복 시간을 주장하지 않는다.",
        "",
        "- [새 원시 metrics.json](../../outputs/patterned_cpu_gpu/wsl/metrics.json)",
        "- [환경 및 장치 증거](../../outputs/patterned_cpu_gpu/wsl/environment.json)",
        "- [동일 조건 반복 측정 manifest](../../outputs/patterned_cpu_gpu/wsl/timing_jobs.json)",
        "- [Linux pip 패키지 고정 목록](../../outputs/patterned_cpu_gpu/wsl/linux_requirements_freeze.txt)",
        "- [Meep conda explicit 목록](../../outputs/patterned_cpu_gpu/wsl/meep_conda_explicit.txt), [fdtdz conda explicit 목록](../../outputs/patterned_cpu_gpu/wsl/fdtdz_conda_explicit.txt)",
        "- [측정 스크립트](../../scripts/benchmark_wsl_solvers.py), [native Meep](../../scripts/benchmark_meep_reference.py)",
        "- [이전 Windows 검토](gpu-execution-audit.md) (과거 환경 기록)",
        "",
    ]
    if success:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for r in highest:
            for i, ax in enumerate(axs):
                ax.plot(
                    np.array(r["wavelength_um"]) * 1000,
                    np.asarray(r["rta"])[:, i],
                    "o-",
                    label=(
                        f"fdtdx dx={r['params']['grid_spacing']}, t={fdtdx_duration(r)}"
                        if r["solver"] == "fdtdx"
                        else f"{r['solver']} N={r['order']}"
                    ),
                )
        if best:
            for i, ax in enumerate(axs):
                ax.plot(
                    [s["wavelength_um"] * 1000 for s in best["spectra"]],
                    np.asarray(ref_values)[:, i],
                    "k--",
                    label=f"Meep r={best['resolution']}",
                )
        for ax, title in zip(axs, ["R", "T", "A (residual)"]):
            ax.set(xlabel="Wavelength (nm)", ylabel=title)
            ax.grid(alpha=0.2)
        axs[0].legend(fontsize=6)
        fig.savefig(OUT / "spectra.png", dpi=170)
        plt.close(fig)
        lines += ["![Measured spectra](../../outputs/patterned_cpu_gpu/wsl/spectra.png)", ""]
        if paired:
            fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
            x = np.arange(len(paired))
            ax.bar(x - 0.2, [a["warm_median_s"] for a, b in paired], 0.4, label="CPU")
            ax.bar(x + 0.2, [b["warm_median_s"] for a, b in paired], 0.4, label="GPU requested")
            ax.set_xticks(x, [a["solver"] for a, b in paired])
            ax.set(ylabel="Warm median seconds / 3 wavelengths", yscale="log")
            ax.legend()
            fig.savefig(OUT / "performance.png", dpi=170)
            plt.close(fig)
            lines += [
                "![Measured performance](../../outputs/patterned_cpu_gpu/wsl/performance.png)",
                "",
            ]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for name, device in sorted({(r["solver"], r["device"]) for r in rcwa}):
            group = {
                r["order"]: r
                for r in rcwa
                if r["solver"] == name and r["device"] == device and r["grid"] == 128
            }
            if group:
                orders = sorted(group)
                axs[0].plot(
                    orders, [group[n]["rta"][2][0] for n in orders], "o-", label=f"{name}/{device}"
                )
        for wl in [0.45, 0.55, 0.65]:
            g = [
                r
                for r in references
                if meep_complete(r) and best and meep_protocol(r) == meep_protocol(best)
            ]
            if g:
                by_resolution = {}
                for run in sorted(g, key=lambda r: r["duration"]):
                    by_resolution[run["resolution"]] = run
                g = [by_resolution[n] for n in sorted(by_resolution)]
                axs[1].plot(
                    [r["resolution"] for r in g],
                    [
                        next(s["rta"][0] for s in r["spectra"] if s["wavelength_um"] == wl)
                        for r in g
                    ],
                    "o-",
                    label=f"{wl * 1000:.0f} nm",
                )
        axs[0].set(xlabel="RCWA order per axis", ylabel="R at 650 nm, grid 128")
        axs[1].set(xlabel="Meep points / um", ylabel="Reflectance")
        for ax in axs:
            ax.grid(alpha=0.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
        fig.savefig(OUT / "convergence.png", dpi=170)
        plt.close(fig)
        lines += [
            "![Measured convergence](../../outputs/patterned_cpu_gpu/wsl/convergence.png)",
            "",
        ]
    if REPORT.exists() and not (OUT / "windows_report_before_wsl.md").exists():
        (OUT / "windows_report_before_wsl.md").write_bytes(REPORT.read_bytes())
    (OUT / "metrics.json").write_text(json.dumps(manifest, indent=2, default=str))
    portable = ROOT / "docs/public/reports/patterned-cpu-gpu-meep-wsl"
    portable.mkdir(parents=True, exist_ok=True)
    for name in PORTABLE_ASSETS:
        if (OUT / name).exists():
            shutil.copy2(OUT / name, portable / name)
    body = "\n".join(lines) + "\n"
    if closeout:
        notice = (
            "사용자의 추가 확장 중단 요청에 따라 측정을 종료했다. 완료된 실측은 보존하고 "
            "실행 중이던 추가 계산은 stopped_by_scope로 기록했다. 대기·중단 데이터는 측정값으로 대체하지 않았다. "
            f"Meep 기준 수렴 판정은 {'통과' if accepted else '미확정'}이며, 인접 최고 완료 격자의 최대 변화는 {grid_delta}이다. "
            "수렴 미확정 상태의 Meep 대비 차이는 잠정 비교로만 해석한다. "
            "N=9 가속률만으로 정확한 솔버를 추천하지 않는다. 추가 계산 프로세스는 모두 종료했다.\n\n"
        )
        first, rest = body.split("\n", 1)
        body = first + "\n\n" + notice + rest
        body = body.replace("미측정 / 진행 중", "미측정 (측정 종료)")
    for name in PORTABLE_ASSETS:
        body = body.replace(
            f"../../outputs/patterned_cpu_gpu/wsl/{name}",
            f"../public/reports/patterned-cpu-gpu-meep-wsl/{name}",
        )
    REPORT.write_text(body)
    print(REPORT)


if __name__ == "__main__":
    write_report()
