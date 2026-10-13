#!/usr/bin/env python3
"""Reproducible end-to-end BF16 attention decode benchmark.

All methods receive identical contiguous BHSD inputs. Each measured sample is
one complete Python call followed by device synchronization, timed with
perf_counter_ns. The synchronization before the call is outside the interval.
FA2's per-call layout conversion is included. Compilation, imports, input
creation, and reference calculation occur outside the timed interval.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from attention_decode.reference import ATOL, RTOL, error_metrics, fa2, get_flash_attn_func, reference, sdpa
from attention_decode.workloads import LENGTHS, all_workloads, make_inputs


FIELDS = (
    "label", "shape", "batch", "query_heads", "kv_heads", "head_dim", "sequence_length",
    "method", "status", "correct", "median_us", "p10_us", "p90_us", "min_us", "max_us",
    "std_us", "cv_pct", "repeat_median_min_us", "repeat_median_max_us", "max_abs_error",
    "max_rel_error", "kv_bytes", "warmup", "samples", "repeats", "seed", "error",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark.csv"))
    parser.add_argument("--metadata", type=Path, help="Default: OUTPUT with .json suffix")
    parser.add_argument("--baseline-only", action="store_true", help="Do not import the custom CUDA implementation")
    parser.add_argument("--label", help="Candidate identifier; default: baseline with --baseline-only, otherwise tc_qk_packed32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lengths", type=int, nargs="+", choices=LENGTHS, default=list(LENGTHS), help="Diagnostic subset only; omit for the full contract")
    args = parser.parse_args()
    if args.label is None:
        args.label = "baseline" if args.baseline_only else "tc_qk_packed32"
    if args.warmup < 1 or args.samples < 2 or args.repeats < 1:
        parser.error("warmup >= 1, samples >= 2, and repeats >= 1 are required")
    return args


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def measure(function, inputs, args, device):
    samples = []
    repeat_medians = []
    for _ in range(args.repeats):
        for _ in range(args.warmup):
            function(*inputs)
        torch.cuda.synchronize(device)
        one_repeat = []
        for _ in range(args.samples):
            torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            result = function(*inputs)
            torch.cuda.synchronize(device)
            elapsed_us = (time.perf_counter_ns() - start) / 1000.0
            one_repeat.append(elapsed_us)
            del result
        samples.extend(one_repeat)
        repeat_medians.append(statistics.median(one_repeat))
    mean = statistics.mean(samples)
    std = statistics.pstdev(samples)
    return {
        "median_us": statistics.median(samples),
        "p10_us": quantile(samples, 0.1),
        "p90_us": quantile(samples, 0.9),
        "min_us": min(samples),
        "max_us": max(samples),
        "std_us": std,
        "cv_pct": 100.0 * std / mean,
        "repeat_median_min_us": min(repeat_medians),
        "repeat_median_max_us": max(repeat_medians),
    }, {"samples_us": samples, "repeat_medians_us": repeat_medians}


def optional_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "error": str(error)}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_fingerprints():
    """Hash code used by this run without importing or executing the extension."""
    sources = (
        "csrc/decode.cu", "csrc/binding.cpp", "attention_decode/__init__.py",
        "attention_decode/reference.py", "benchmarks/bench_attention_decode.py",
    )
    hashes = {}
    for relative in sources:
        path = ROOT / relative
        # Baseline-only runs are also supported before the kernel is created.
        if not path.is_file():
            hashes[relative] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative] = digest.hexdigest()
    return hashes


def metadata(args, device):
    props = torch.cuda.get_device_properties(device)
    git_status = optional_command(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flash_attn": package_version("flash-attn"),
        "device": str(device),
        "gpu": props.name,
        "compute_capability": [props.major, props.minor],
        "multiprocessor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "nvidia_smi": optional_command(["nvidia-smi", "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit", "--format=csv"]),
        "nvcc": optional_command(["nvcc", "--version"]),
        "git_revision": optional_command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_status": git_status,
        "git_dirty": bool(git_status["stdout"]) if git_status.get("returncode") == 0 else None,
        "source_sha256": source_fingerprints(),
        "timing": "perf_counter_ns: complete Python call + following CUDA device synchronization; pre-call synchronization excluded",
        "fa2_layout_adapter": "BHSD -> BSHD transpose views (last dimension contiguous), flash_attn_func(causal=False), BHSD output; all view operations timed, no forced KV copies",
        "reference": "FP32 grouped QK/softmax/PV with TF32 disabled, then BF16 rounding",
        "atol": ATOL,
        "rtol": RTOL,
        "relative_error_denominator_floor": 1e-6,
        "warmup_per_repeat": args.warmup,
        "samples_per_repeat": args.samples,
        "repeats": args.repeats,
        "seed": args.seed,
        "lengths": args.lengths,
        "full_contract": set(args.lengths) == set(LENGTHS),
        "method_order": "rotated by workload index to distribute order effects",
        "label": args.label,
        "raw_measurements": {},
    }


def summarize(rows, workload_count):
    summary = {"workload_count": workload_count, "methods": {}, "speedups_t_ge_128": {}}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            "ok": sum(row["status"] == "ok" for row in selected),
            "total": len(selected),
            "failures": [{"shape": row["shape"], "status": row["status"], "error": row.get("error", "")} for row in selected if row["status"] != "ok"],
        }
    custom = {row["shape"]: row for row in rows if row["method"] == "custom" and row["sequence_length"] >= 128}
    for baseline in ("sdpa", "fa2"):
        matching = {row["shape"]: row for row in rows if row["method"] == baseline and row["sequence_length"] >= 128}
        pairs = [(matching[key], row) for key, row in custom.items() if row["status"] == "ok" and matching[key]["status"] == "ok"]
        if custom:
            ratios = [base["median_us"] / candidate["median_us"] for base, candidate in pairs]
            summary["speedups_t_ge_128"][baseline] = {
                "geomean": math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
                "matched_shapes": len(pairs),
                "expected_shapes": 40,
                "selected_shapes": len(custom),
                "subset_complete": len(pairs) == len(custom),
                "complete": len(pairs) == 40,
                "regressions": [{"shape": candidate["shape"], "speedup": base["median_us"] / candidate["median_us"]} for base, candidate in pairs if candidate["median_us"] > base["median_us"]],
            }
    return summary


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. On this WSL host run the command outside the sandbox with llm-bench Python.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    report = metadata(args, device)
    methods = {"sdpa": sdpa, "fa2": fa2}
    unavailable = {}
    try:
        get_flash_attn_func()
    except Exception as error:
        unavailable["fa2"] = f"{type(error).__name__}: {error}"
    if not args.baseline_only:
        from attention_decode import attention_decode

        methods["custom"] = attention_decode
    workloads = [(index, shape) for index, shape in enumerate(all_workloads()) if shape.sequence_length in args.lengths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(json.dumps({key: report[key] for key in ("gpu", "compute_capability", "torch", "torch_cuda", "flash_attn", "timing")}, ensure_ascii=False), flush=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, shape in workloads:
            base_row = {"label": args.label, "shape": shape.key, "batch": shape.batch, "query_heads": shape.query_heads, "kv_heads": shape.kv_heads, "head_dim": shape.head_dim, "sequence_length": shape.sequence_length, "kv_bytes": shape.kv_bytes, "warmup": args.warmup, "samples": args.samples, "repeats": args.repeats, "seed": args.seed + index}
            inputs = expected = None
            setup_error = None
            try:
                inputs = make_inputs(shape, device=device, seed=args.seed + index)
                expected = reference(*inputs)
                torch.cuda.synchronize(device)
            except Exception as error:
                setup_error = ("oom" if isinstance(error, torch.OutOfMemoryError) else "reference_error", f"{type(error).__name__}: {error}")
            order = list(methods)
            offset = index % len(order)
            for method in order[offset:] + order[:offset]:
                row = {**base_row, "method": method, "status": "ok", "error": ""}
                if setup_error:
                    row.update(status=setup_error[0], error=setup_error[1], correct=False)
                elif method in unavailable:
                    row.update(status="unavailable", error=unavailable[method], correct=False)
                else:
                    output = None
                    try:
                        output = methods[method](*inputs)
                        torch.cuda.synchronize(device)
                        row.update(error_metrics(output, expected))
                        del output
                        output = None
                        if not row["correct"]:
                            row["status"] = "incorrect"
                        else:
                            timing, raw = measure(methods[method], inputs, args, device)
                            row.update(timing)
                            report["raw_measurements"][f"{shape.key}/{method}"] = raw
                    except Exception as error:
                        row.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "runtime_error", error=f"{type(error).__name__}: {error}", correct=False)
                    finally:
                        del output
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                latency = f"{row['median_us']:.3f} us" if "median_us" in row else row.get("error", "")
                print(f"{shape.key:32s} {method:6s} {row['status']:15s} {latency}", flush=True)
            del inputs, expected
            torch.cuda.empty_cache()
    report["summary"] = summarize(rows, len(workloads))
    report["nvidia_smi_after"] = optional_command(["nvidia-smi", "--query-gpu=pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw", "--format=csv"])
    with metadata_path.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"CSV: {args.output}; metadata and raw samples: {metadata_path}", flush=True)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
