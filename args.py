"""
CLI-friendly tool for DMT2 risk assessment model training and inference.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import argparse


class Training:
    def build_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="DMT2 risk prediction model training."
        )

        parser.add_argument(
            "--datafiles",
            type=str,
            nargs="+",
            default=[],
            help="File paths for data to import."
        )
        parser.add_argument(
            "--cache_path",
            type=str,
            default="./dataset_cache.pkl",
            help="Optional cache path for faster load times."
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Optional flag for verbose messages to stdout."
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Optional random seed. Default 42."
        )

        return parser.parse_args()
