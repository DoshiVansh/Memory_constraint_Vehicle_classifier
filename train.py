"""Compatibility entry point for the supported student training workflow."""

import argparse

from train_distillation import train_distillation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distillation.yaml")
    args = parser.parse_args()
    train_distillation(args.config)


if __name__ == "__main__":
    main()
