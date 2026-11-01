"""Utility script to robustly configure ~/.kube/config from KUBE_CONFIG_DATA.

Handles raw YAML kubeconfig strings, standard base64 encodings, and base64 strings
with arbitrary whitespace, newlines, or CRLF introduced by web forms or CLI exports.
"""

import base64
import os
from pathlib import Path


def setup_kubeconfig() -> None:
    raw_data = os.environ.get("KUBE_CONFIG_DATA", "").strip()
    if not raw_data:
        raise ValueError("Environment variable KUBE_CONFIG_DATA is empty or not set.")

    kube_dir = Path.home() / ".kube"
    kube_dir.mkdir(parents=True, exist_ok=True)
    kube_config = kube_dir / "config"

    # Check if raw_data is already plaintext YAML
    if raw_data.startswith("apiVersion:") or "clusters:" in raw_data:
        print("Detected plaintext YAML kubeconfig data.")
        content = raw_data.encode("utf-8")
    else:
        print("Decoding base64-encoded kubeconfig data...")
        try:
            # Python's base64 ignores standard whitespace, but remove internal newlines/carriage returns
            clean_b64 = "".join(raw_data.split())
            missing_padding = len(clean_b64) % 4
            if missing_padding:
                clean_b64 += "=" * (4 - missing_padding)
            content = base64.b64decode(clean_b64, validate=False)
        except Exception as exc:
            raise ValueError(f"Failed to decode KUBE_CONFIG_DATA as base64: {exc}") from exc

    # Sanity check: must look like a valid Kubernetes config
    text_sample = content[:300].decode("utf-8", errors="ignore")
    if "apiVersion" not in text_sample and "clusters" not in text_sample and "users" not in text_sample:
        raise ValueError(
            "Decoded kubeconfig content does not appear to be a valid Kubernetes configuration (missing apiVersion/clusters)."
        )

    kube_config.write_bytes(content)
    try:
        os.chmod(kube_config, 0o600)
    except OSError:
        pass

    print(f"Successfully configured {kube_config} ({len(content)} bytes, mode 0600).")


if __name__ == "__main__":
    setup_kubeconfig()
