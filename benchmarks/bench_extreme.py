#!/usr/bin/env python3
"""Independent variant comparisons with palindromic measurement blocks.

E2E retains the original call-plus-device-sync boundary. Event batches include
host supply gaps. Graph captures many complete forwards, then measures one
replay; its static allocation and input-address semantics are reported apart.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from attention_decode.reference import ATOL, RTOL, error_metrics, fa2, get_flash_attn_func, reference, sdpa
from attention_decode.workloads import LENGTHS, all_workloads, make_inputs
from scripts.extreme_variants import file_hash, load_variant, optional_command, verify_independent


FIELDS = ("label", "shape", "batch", "query_heads", "kv_heads", "head_dim", "sequence_length",
          "cache_mode", "mode", "method", "method_label", "status", "correct", "median_us",
          "p10_us", "p90_us", "min_us", "max_us", "cv_pct", "samples", "blocks",
          "block_median_min_us", "block_median_max_us", "batch_calls", "rotation_slots",
          "kv_working_set_bytes", "peak_allocated_delta_bytes", "max_abs_error", "max_rel_error", "error")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New CSV path; raw JSON uses the same stem")
    parser.add_argument("--label", default="extreme")
    parser.add_argument("--start-label", default="start")
    parser.add_argument("--incumbent-label", default="incumbent")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--modes", nargs="+", choices=("graph", "event", "e2e"), default=["graph", "event", "e2e"])
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batch-calls", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", choices=LENGTHS, default=list(LENGTHS))
    parser.add_argument("--shapes", nargs="+", help="Exact Workload.key values, further filtered by --lengths")
    parser.add_argument("--rotation-bytes", type=int, default=0,
                        help="KV ring target bytes, 0 for fixed hot inputs; one ring used symmetrically by every method")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.blocks, args.samples, args.batch_calls, args.warmup) < 1 or args.rotation_bytes < 0:
        parser.error("blocks, samples, batch-calls and warmup must be positive; rotation-bytes must be nonnegative")
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must be unique")
    known = {shape.key for shape in all_workloads()}
    if args.shapes and (set(args.shapes) - known):
        parser.error(f"Unknown shapes: {sorted(set(args.shapes) - known)}")
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must have a .csv suffix")
    if not any(shape.sequence_length in args.lengths and (not args.shapes or shape.key in args.shapes)
               for shape in all_workloads()):
        parser.error("No selected workloads")
    return args


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def statistics_for(values):
    mean = statistics.mean(values)
    return {"median_us": statistics.median(values), "p10_us": quantile(values, .1),
            "p90_us": quantile(values, .9), "min_us": min(values), "max_us": max(values),
            "cv_pct": 100 * statistics.pstdev(values) / mean if mean else 0,
            "samples": len(values)}


def palindromic_order(methods, block):
    offset = block % len(methods)
    forward = methods[offset:] + methods[:offset]
    return forward + list(reversed(forward))


def batch_count(shape, requested, slots):
    # Limit useful KV processed in a batch to about 512 MiB, but always cover a
    # whole ring. This is identical for all methods and chosen before timing.
    bounded = min(requested, max(1, (512 * 1024 * 1024) // shape.kv_bytes))
    return max(slots, (bounded // slots) * slots)


def atomic_json(path, report):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_csv(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def gpu_environment():
    return {"gpu": optional_command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used", "--format=csv"]),
            "processes": optional_command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"])}


class Timer:
    def __init__(self, function, inputs, mode, calls, warmup, device):
        self.function, self.inputs, self.mode = function, inputs, mode
        self.calls, self.device = calls, device
        self.graph = None
        self.outputs = []
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            for i in range(max(warmup, len(inputs))):
                function(*inputs[i % len(inputs)])
        self.stream.synchronize()
        if mode == "graph":
            self.graph = torch.cuda.CUDAGraph()
            # Keep the last output for each ring slot for correctness checking;
            # transient earlier outputs/partials can be reused by the graph pool.
            self.outputs = [None] * len(inputs)
            with torch.cuda.graph(self.graph, stream=self.stream):
                for i in range(calls):
                    slot = i % len(inputs)
                    self.outputs[slot] = function(*inputs[slot])
            with torch.cuda.stream(self.stream):
                self.graph.replay()
            self.stream.synchronize()

    def check_graph(self, expected):
        if self.graph is None:
            return None
        errors = [error_metrics(output, target) for output, target in zip(self.outputs, expected)]
        if not all(item["correct"] for item in errors):
            raise RuntimeError(f"Graph replay correctness failed: {errors}")
        # Input addresses are static but content must remain live. Updating V
        # tests replay execution and rules out a accidentally frozen result.
        saved = self.inputs[0][2].clone()
        self.inputs[0][2].zero_()
        torch.cuda.synchronize(self.device)
        try:
            with torch.cuda.stream(self.stream):
                self.graph.replay()
            self.stream.synchronize()
            if not bool((self.outputs[0] == 0).all()):
                raise RuntimeError("Graph did not observe updated static V input")
        finally:
            self.inputs[0][2].copy_(saved)
            torch.cuda.synchronize(self.device)
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        self.stream.synchronize()
        restored = error_metrics(self.outputs[0], expected[0])
        if not restored["correct"]:
            raise RuntimeError(f"Graph replay after input restoration failed: {restored}")
        return {"all_slots_correct": True, "input_mutation_and_restore_correct": True,
                "slots": len(errors)}

    def warm(self, count):
        with torch.cuda.stream(self.stream):
            if self.mode == "graph":
                self.graph.replay()
            else:
                for i in range(max(count, len(self.inputs))):
                    self.function(*self.inputs[i % len(self.inputs)])
        self.stream.synchronize()

    def sample(self, sample_index):
        if self.mode == "e2e":
            torch.cuda.synchronize(self.device)
            with torch.cuda.stream(self.stream):
                start = time.perf_counter_ns()
                output = self.function(*self.inputs[sample_index % len(self.inputs)])
                torch.cuda.synchronize(self.device)
                elapsed = (time.perf_counter_ns() - start) / 1000
            del output
            return elapsed
        with torch.cuda.stream(self.stream):
            self.start.record()
            if self.mode == "graph":
                self.graph.replay()
            else:
                for i in range(self.calls):
                    output = self.function(*self.inputs[i % len(self.inputs)])
            self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000 / self.calls


def rows_for_record(args, shape, record, labels):
    rows = []
    for mode in args.modes:
        for method in labels:
            status = record["methods"][method]
            positions = [position for block in record.get("measurements", {}).get(mode, [])
                         for position in block["positions"] if position["method"] == method]
            samples = [value for position in positions for value in position["samples_us"]]
            failure = record.get("mode_errors", {}).get(mode, {}).get(method)
            row = {"label": args.label, "shape": shape.key, "batch": shape.batch,
                   "query_heads": shape.query_heads, "kv_heads": shape.kv_heads,
                   "head_dim": shape.head_dim, "sequence_length": shape.sequence_length,
                   "cache_mode": "rotation" if args.rotation_bytes else "hot", "mode": mode,
                   "method": method, "method_label": labels[method],
                   "status": "error" if failure else status["status"],
                   "correct": status.get("correct", False), "blocks": len(record.get("measurements", {}).get(mode, [])),
                   "batch_calls": 1 if mode == "e2e" else record["batch_calls"],
                   "rotation_slots": record["rotation_slots"],
                   "kv_working_set_bytes": record["kv_working_set_bytes"],
                   "peak_allocated_delta_bytes": status.get("peak_allocated_delta_bytes"),
                   "max_abs_error": status.get("max_abs_error"), "max_rel_error": status.get("max_rel_error"),
                   "error": failure or status.get("error", "")}
            if samples:
                row.update(statistics_for(samples))
                medians = [block["method_medians_us"][method]
                           for block in record["measurements"][mode] if method in block["method_medians_us"]]
                row.update(block_median_min_us=min(medians), block_median_max_us=max(medians))
            elif row["status"] == "ok":
                row["status"] = "pending"
            rows.append(row)
    return rows


def paired_summary(report):
    result = {}
    for mode in report["modes"]:
        result[mode] = {}
        for base in report["method_labels"]:
            if base == "candidate":
                continue
            ratios = []
            per_shape = []
            for key, record in report["workloads"].items():
                if record["shape"]["sequence_length"] < 128:
                    continue
                values = [block["candidate_speedups"][base] for block in record.get("measurements", {}).get(mode, [])
                          if base in block["candidate_speedups"]]
                if len(values) != report["blocks_requested"]:
                    continue
                ratio = math.exp(statistics.mean(map(math.log, values)))
                ratios.append(ratio)
                per_shape.append({"shape": key, "speedup": ratio, "block_speedups": values})
            result[mode][base] = {"geomean": math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
                                  "matched_shapes": len(ratios), "expected_shapes": 40,
                                  "complete": len(ratios) == 40, "per_shape": per_shape,
                                  "regressions": [item for item in per_shape if item["speedup"] < 1],
                                  "interpretation": "Descriptive paired block ratios; no independent-round confidence or promotion claim"}
    return result


@torch.inference_mode()
def main():
    args = parse_args()
    metadata_path = args.output.with_suffix(".json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve both paths before any compilation, so old experimental data cannot
    # be overwritten even when two invocations race.
    if args.output.exists() or metadata_path.exists():
        raise SystemExit(f"Refusing to overwrite existing experiment: {args.output} / {metadata_path}")
    args.output.touch(exist_ok=False)
    metadata_path.touch(exist_ok=False)
    shapes = [(index, shape) for index, shape in enumerate(all_workloads())
              if shape.sequence_length in args.lengths and (not args.shapes or shape.key in args.shapes)]
    if not shapes:
        raise SystemExit("No selected workloads")
    report = {"schema_version": 1, "status": "initializing", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "command": [sys.executable, *sys.argv], "python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "script_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in
                                (Path(__file__), ROOT / "scripts/extreme_variants.py", ROOT / "attention_decode/reference.py",
                                 ROOT / "attention_decode/workloads.py")},
              "git_revision": optional_command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
              "git_status": optional_command(["git", "-C", str(ROOT), "status", "--porcelain"]),
              "modes": args.modes, "blocks_requested": args.blocks, "samples_per_position": args.samples,
              "warmup": args.warmup, "batch_calls_requested": args.batch_calls,
              "rotation_bytes_requested": args.rotation_bytes, "seed": args.seed,
              "selected_shapes": [shape.key for _, shape in shapes], "full_contract": len(shapes) == 56,
              "timing": {"e2e": "perf_counter_ns: full public Python API call + following device sync; pre-sync excluded",
                         "event": "same-stream CUDA events around N full API calls / N; includes host supply gaps",
                         "graph": "same-stream CUDA events around one replay of a graph containing N full forwards / N; capture excluded; capture allocation reuse and static input addresses; not ordinary API allocation cost"},
              "statistics": "One palindromic block is the comparison unit; within-position samples and N forwards are correlated, not independent evidence. No samples discarded.",
              "reference": "TF32-disabled FP32 grouped attention rounded BF16", "atol": ATOL, "rtol": RTOL,
              "relative_error_denominator_floor": 1e-6, "variants": {}, "workloads": {}, "method_labels": {}}
    atomic_json(metadata_path, report)
    rows = []
    variants = []
    device = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        report["device"] = {"name": props.name, "capability": [props.major, props.minor],
                            "sm_count": props.multi_processor_count, "total_memory_bytes": props.total_memory,
                            "l2_cache_size_bytes": getattr(props, "L2_cache_size", None),
                            "l2_source": "torch.cuda.get_device_properties.L2_cache_size when exposed; null means unavailable"}
        report["environment_before"] = gpu_environment()
        specifications = [("start", args.start, args.start_label)]
        if args.incumbent:
            specifications.append(("incumbent", args.incumbent, args.incumbent_label))
        specifications.append(("candidate", args.candidate, args.candidate_label))
        methods = {}
        labels = {}
        for role, root, label in specifications:
            variant = load_variant(root, role, label=label)
            variants.append(variant)
            methods[role], labels[role] = variant.function, label
            report["variants"][role] = variant.manifest
            atomic_json(metadata_path, report)
        verify_independent(variants)
        report["independent_identity_verified_before"] = True
        unavailable = {}
        if args.baselines:
            methods.update(sdpa=sdpa, fa2=fa2)
            labels.update(sdpa="sdpa", fa2="fa2")
            try:
                get_flash_attn_func()
                report["flash_attn"] = importlib.metadata.version("flash-attn")
            except Exception as error:
                unavailable["fa2"] = f"{type(error).__name__}: {error}"
        report["method_labels"] = labels
        report["incumbent_alias"] = None if args.incumbent else "start"
        report["status"] = "running"
        atomic_json(metadata_path, report)
        for index, shape in shapes:
            slots = max(1, math.ceil(args.rotation_bytes / shape.kv_bytes))
            calls = batch_count(shape, args.batch_calls, slots)
            inputs = [make_inputs(shape, device=device, seed=args.seed + index + slot * 10000) for slot in range(slots)]
            expected = [reference(*item) for item in inputs]
            torch.cuda.synchronize(device)
            record = {"shape": shape.__dict__, "rotation_slots": slots,
                      "kv_working_set_bytes": slots * shape.kv_bytes,
                      "input_allocated_bytes": sum(x.numel() * x.element_size() for item in inputs for x in item),
                      "batch_calls": calls, "methods": {}, "measurements": {}, "mode_errors": {},
                      "graph_validation": {}, "environment_before": gpu_environment()}
            report["workloads"][shape.key] = record
            for method, function in methods.items():
                status = {"status": "ok", "correct": True, "error": ""}
                try:
                    if method in unavailable:
                        raise RuntimeError(unavailable[method])
                    errors = []
                    for item, target in zip(inputs, expected):
                        output = function(*item)
                        torch.cuda.synchronize(device)
                        errors.append(error_metrics(output, target))
                        del output
                    status["correct"] = all(item["correct"] for item in errors)
                    # Preserve failure diagnostics while keeping incremental
                    # JSON valid even when a candidate returns NaN or Inf.
                    for field in ("max_abs_error", "max_rel_error"):
                        finite = [item[field] for item in errors
                                  if isinstance(item.get(field), (float, int)) and math.isfinite(item[field])]
                        status[field] = max(finite) if len(finite) == len(errors) else None
                    if not status["correct"]:
                        raise RuntimeError(f"Correctness failed: {errors}")
                    torch.cuda.reset_peak_memory_stats(device)
                    allocated = torch.cuda.memory_allocated(device)
                    output = function(*inputs[0])
                    torch.cuda.synchronize(device)
                    status["peak_allocated_delta_bytes"] = torch.cuda.max_memory_allocated(device) - allocated
                    del output
                except Exception as error:
                    status.update(status="error", correct=False, error=f"{type(error).__name__}: {error}")
                record["methods"][method] = status
            atomic_json(metadata_path, report)
            for mode in args.modes:
                timers = {}
                record["measurements"][mode] = []
                record["mode_errors"][mode] = {}
                for method, function in methods.items():
                    if record["methods"][method]["status"] != "ok":
                        continue
                    try:
                        timer = Timer(function, inputs, mode, calls, args.warmup, device)
                        if mode == "graph":
                            record["graph_validation"][method] = timer.check_graph(expected)
                        timers[method] = timer
                    except Exception as error:
                        record["mode_errors"][mode][method] = f"{type(error).__name__}: {error}"
                for block_index in range(args.blocks):
                    order = palindromic_order(list(timers), block_index + index) if timers else []
                    block = {"index": block_index, "order": order, "positions": [],
                             "method_medians_us": {}, "method_log_center_us": {}, "candidate_speedups": {}}
                    for position, method in enumerate(order):
                        timer = timers[method]
                        timer.warm(args.warmup)
                        values = [timer.sample(sample + block_index * args.samples) for sample in range(args.samples)]
                        block["positions"].append({"position": position, "method": method, "samples_us": values,
                                                   **statistics_for(values)})
                    for method in timers:
                        positions = [entry for entry in block["positions"] if entry["method"] == method]
                        block["method_medians_us"][method] = statistics.median(
                            [value for entry in positions for value in entry["samples_us"]])
                        block["method_log_center_us"][method] = math.exp(statistics.mean(
                            math.log(entry["median_us"]) for entry in positions))
                    if "candidate" in block["method_log_center_us"]:
                        candidate = block["method_log_center_us"]["candidate"]
                        block["candidate_speedups"] = {method: value / candidate for method, value in block["method_log_center_us"].items()
                                                        if method != "candidate"}
                    record["measurements"][mode].append(block)
                    report["summary"] = paired_summary(report)
                    atomic_json(metadata_path, report)
                    print(f"{shape.key} {mode} block {block_index + 1}/{args.blocks}: "
                          + ", ".join(f"{key}={value:.3f}us" for key, value in block["method_log_center_us"].items()), flush=True)
                del timers
                if "timer" in locals():
                    del timer
                gc.collect()
                torch.cuda.synchronize(device)
            record["environment_after"] = gpu_environment()
            rows.extend(rows_for_record(args, shape, record, labels))
            save_csv(args.output, rows)
            atomic_json(metadata_path, report)
            del inputs, expected
            gc.collect()
            torch.cuda.empty_cache()
        verify_independent(variants)
        report["independent_identity_verified_after"] = True
        report["status"] = "complete" if all(row["status"] == "ok" for row in rows) else "completed_with_errors"
    except BaseException as error:
        report["status"] = "failed"
        report["fatal_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        atomic_json(metadata_path, report)
        raise
    finally:
        if device is not None:
            report["environment_after"] = gpu_environment()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["summary"] = paired_summary(report)
        atomic_json(metadata_path, report)
    print(json.dumps({"status": report["status"], "csv": str(args.output), "json": str(metadata_path)}, indent=2), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
