from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate samples with DDPM or DDIM (stub until sampling is implemented)."
    )
    parser.add_argument(
        "--method",
        choices=("ddpm", "ddim"),
        default="ddim",
        help="Sampler to run once implemented.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained weights.",
    )
    args = parser.parse_args()
    print(
        "Stub: no sampling yet. Implement find_delft_assignment.sampling and wire this CLI.",
        args,
    )


if __name__ == "__main__":
    main()
