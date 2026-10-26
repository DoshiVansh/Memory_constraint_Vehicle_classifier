"""Run a bounded-concurrency prediction load and report client/service metrics."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zlib


def png_sample():
    """Generate a deterministic 32x32 RGB PNG using only the standard library."""

    def chunk(kind, data):
        payload = kind + data
        return len(data).to_bytes(4, "big") + payload + (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")

    width = height = 32
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(((x * 7) % 256, (y * 7) % 256, ((x + y) * 3) % 256))
    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes((8, 2, 0, 0, 0))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def multipart_body(image_bytes):
    boundary = "vehicle-classifier-load-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="load-sample.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def fetch(url, data=None, headers=None, timeout=30):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
            return response.status, content, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), time.perf_counter() - started


def make_request(base_url, body, content_type):
    status, content, elapsed = fetch(
        f"{base_url}/predict", body, {"Content-Type": content_type}
    )
    valid = False
    try:
        payload = json.loads(content)
        valid = (
            status == 200
            and isinstance(payload.get("predicted_index"), int)
            and payload.get("class_name")
            and payload.get("model_sha256")
        )
    except (ValueError, AttributeError):
        payload = None
    return {"status": status, "elapsed_seconds": elapsed, "valid_prediction": bool(valid), "response": payload}


def read_process_metrics(metrics_url):
    with urllib.request.urlopen(metrics_url, timeout=2) as response:
        exposition = response.read().decode("utf-8", errors="replace")
    values = {}
    for line in exposition.splitlines():
        if line.startswith("process_cpu_seconds_total "):
            values["cpu_seconds"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("process_resident_memory_bytes "):
            values["resident_memory_bytes"] = float(line.rsplit(" ", 1)[1])
    return values


def metric_sampler(metrics_url, stop_event, samples, errors):
    while not stop_event.is_set():
        try:
            samples.append({"time": time.monotonic(), **read_process_metrics(metrics_url)})
        except (OSError, ValueError, urllib.error.URLError) as exc:
            errors.append(type(exc).__name__)
        stop_event.wait(0.5)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def host_memory_bytes():
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty():
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def run_level(base_url, body, content_type, concurrency, request_count):
    metrics_url = f"{base_url}/metrics"
    stop_event = threading.Event()
    resource_samples = []
    resource_errors = []
    sampler = threading.Thread(
        target=metric_sampler,
        args=(metrics_url, stop_event, resource_samples, resource_errors),
        daemon=True,
    )
    sampler.start()
    time.sleep(0.2)
    started = time.perf_counter()
    outcomes = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request, base_url, body, content_type) for _ in range(request_count)]
        for future in as_completed(futures):
            outcomes.append(future.result())
    elapsed = time.perf_counter() - started
    stop_event.set()
    sampler.join(timeout=3)

    latencies = [row["elapsed_seconds"] for row in outcomes]
    successes = sum(row["valid_prediction"] for row in outcomes)
    errors_by_status = {}
    class_distribution = Counter()
    for row in outcomes:
        if row["valid_prediction"]:
            class_distribution[row["response"]["class_name"]] += 1
        else:
            key = str(row["status"])
            errors_by_status[key] = errors_by_status.get(key, 0) + 1

    cpu_samples = [row for row in resource_samples if "cpu_seconds" in row]
    memory_samples = [row["resident_memory_bytes"] for row in resource_samples if "resident_memory_bytes" in row]
    cpu_cores = None
    if len(cpu_samples) >= 2:
        cpu_elapsed = cpu_samples[-1]["time"] - cpu_samples[0]["time"]
        if cpu_elapsed > 0:
            cpu_cores = max(0, cpu_samples[-1]["cpu_seconds"] - cpu_samples[0]["cpu_seconds"]) / cpu_elapsed

    return {
        "concurrency": concurrency,
        "requests": request_count,
        "successes": successes,
        "errors": request_count - successes,
        "error_rate": (request_count - successes) / request_count,
        "error_statuses": errors_by_status,
        "prediction_class_counts": dict(sorted(class_distribution.items())),
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": request_count / elapsed if elapsed else 0,
        "latency_seconds": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies, default=0),
        },
        "service_process": {
            "cpu_cores_average_during_sample": cpu_cores,
            "resident_memory_bytes_start": memory_samples[0] if memory_samples else None,
            "resident_memory_bytes_peak_sampled": max(memory_samples) if memory_samples else None,
            "resident_memory_bytes_end": memory_samples[-1] if memory_samples else None,
            "sample_count": len(resource_samples),
            "sampling_errors": sorted(set(resource_errors)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--requests-per-level", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--service-image-ref", default=os.getenv("SERVICE_IMAGE_REF", "unknown"))
    parser.add_argument("--cluster-context", default=os.getenv("KUBERNETES_CONTEXT", "unknown"))
    args = parser.parse_args()
    if args.requests_per_level < 1 or any(level < 1 for level in args.concurrency):
        parser.error("Concurrency and request count must be positive")

    base_url = args.base_url.rstrip("/")
    status, ready_bytes, _ = fetch(f"{base_url}/health/ready", timeout=5)
    if status != 200:
        raise SystemExit(f"API is not ready (HTTP {status}): {ready_bytes[:300]!r}")
    readiness = json.loads(ready_bytes)
    image_bytes = png_sample()
    body, content_type = multipart_body(image_bytes)
    for _ in range(3):
        result = make_request(base_url, body, content_type)
        if not result["valid_prediction"]:
            raise SystemExit(f"Warm-up prediction failed: {result}")

    results = [
        run_level(base_url, body, content_type, level, args.requests_per_level)
        for level in args.concurrency
    ]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "git_worktree_dirty": git_worktree_dirty(),
        "target": base_url,
        "service_image_ref": args.service_image_ref,
        "cluster_context": args.cluster_context,
        "model_sha256": readiness.get("model_sha256"),
        "workload": {
            "method": "multipart POST /predict",
            "image_format": "PNG",
            "image_bytes": len(image_bytes),
            "warmup_requests": 3,
            "concurrency_levels": args.concurrency,
            "requests_per_level": args.requests_per_level,
        },
        "client_environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "host_memory_bytes": host_memory_bytes(),
        },
        "results": results,
        "interpretation": "Local demo measurements only; do not use as production SLO or capacity claims.",
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
