"""Utility script to substitute environment variables into Kubernetes manifest templates."""

import os
import sys

TEMPLATES = [
    ("k8s/configmap.yaml.tpl", "k8s/configmap.yaml"),
    ("k8s/deployment.yaml.tpl", "k8s/deployment.yaml"),
]

ENV_VARS = ["MODEL_VERSION", "RELEASE_GIT_SHA", "MODEL_SHA256", "IMAGE_REF"]


def render_manifests():
    for tpl_path, out_path in TEMPLATES:
        if not os.path.exists(tpl_path):
            print(f"Skipping missing template: {tpl_path}")
            continue
        with open(tpl_path, "r", encoding="utf-8") as f:
            content = f.read()
        for var in ENV_VARS:
            val = os.environ.get(var, "")
            content = content.replace("${" + var + "}", val).replace("$" + var, val)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Rendered {out_path} from {tpl_path}")


if __name__ == "__main__":
    render_manifests()
