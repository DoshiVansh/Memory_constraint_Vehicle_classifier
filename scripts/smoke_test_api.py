#!/usr/bin/env python3
"""Smoke-test health and image prediction against a running API."""

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
import uuid

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMQ0bABAADMAHm9FcOvAAAAAElFTkSuQmCC"
)


def request(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    ready = False
    for _ in range(60):
        try:
            status, _ = request(base_url + "/health/ready")
            if status == 200:
                ready = True
                break
        except (OSError, TimeoutError):
            pass
        time.sleep(1)
    assert ready, "API did not become ready within 60 seconds"

    for path in ("/health/live", "/health/ready"):
        status, body = request(base_url + path)
        assert status == 200, f"{path}: HTTP {status}: {body!r}"

    boundary = "----vehicle-classifier-" + uuid.uuid4().hex
    multipart = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.png\"\r\n"
        "Content-Type: image/png\r\n\r\n"
    ).encode() + PNG_1X1 + f"\r\n--{boundary}--\r\n".encode()
    status, body = request(
        base_url + "/predict", multipart,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert status == 200, f"valid image: HTTP {status}: {body!r}"
    prediction = json.loads(body)
    assert prediction["predicted_index"] in range(5), prediction
    assert prediction["class_name"] in {"Bus", "Truck", "Car", "Bike", "None"}, prediction
    assert "confidence" in prediction and 0.0 <= prediction["confidence"] <= 1.0, prediction
    assert "probabilities" in prediction and len(prediction["probabilities"]) == 5, prediction

    malformed = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"bad.png\"\r\n"
        "Content-Type: image/png\r\n\r\nnope\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    status, body = request(
        base_url + "/predict", malformed,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert status == 400, f"malformed image: HTTP {status}: {body!r}"
    print("API smoke passed: live, ready, valid prediction, malformed-image rejection")


if __name__ == "__main__":
    main()
