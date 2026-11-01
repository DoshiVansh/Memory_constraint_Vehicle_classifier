"""Verify readiness and a real prediction against a forwarded Kubernetes Service."""

import argparse
import json
import struct
import time
import urllib.error
import urllib.request
import zlib


def tiny_png():
    """Return a valid 2x2 RGB PNG using only the Python standard library."""

    def chunk(kind, data):
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    rows = b"\x00" + b"\x80\x40\x20" * 2
    rows += b"\x00" + b"\x20\x80\x40" * 2
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def request(url, data=None, headers=None, timeout=3):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read()


def multipart_png():
    boundary = "vehicle-classifier-smoke-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="smoke.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + tiny_png() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout_seconds
    last_error = None

    while time.monotonic() < deadline:
        try:
            status, ready_bytes = request(f"{args.base_url}/health/ready")
            ready = json.loads(ready_bytes)
            if status == 200 and ready.get("status") == "ready":
                break
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
        time.sleep(2)
    else:
        raise SystemExit(f"Readiness did not succeed within {args.timeout_seconds}s: {last_error}")

    body, content_type = multipart_png()
    status, prediction_bytes = request(
        f"{args.base_url}/predict", body, {"Content-Type": content_type}, timeout=30
    )
    prediction = json.loads(prediction_bytes)
    if status != 200:
        raise SystemExit(f"Prediction smoke returned HTTP {status}: {prediction}")
    if not isinstance(prediction.get("predicted_index"), int):
        raise SystemExit(f"Prediction response has no integer predicted_index: {prediction}")
    if not prediction.get("class_name") or not prediction.get("model_sha256"):
        raise SystemExit(f"Prediction response is missing model fields: {prediction}")
    if ready.get("model_sha256") != prediction.get("model_sha256"):
        raise SystemExit("Readiness and prediction report different model checksums")
    print(json.dumps({"readiness": ready, "prediction": prediction}, sort_keys=True))


if __name__ == "__main__":
    main()
