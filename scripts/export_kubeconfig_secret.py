"""Helper script to display the base64-encoded KUBECONFIG for GitHub Secrets."""

import base64
from pathlib import Path


def main():
    kubeconfig_path = Path.home() / ".kube" / "config"
    if not kubeconfig_path.exists():
        print(f"Error: Kubeconfig not found at {kubeconfig_path}")
        return
    encoded = base64.b64encode(kubeconfig_path.read_bytes()).decode("utf-8")
    print("=" * 60)
    print("Base64 Encoded Kubeconfig for GitHub Secret 'KUBE_CONFIG_DATA'")
    print("=" * 60)
    print(encoded)
    print("=" * 60)


if __name__ == "__main__":
    main()
