"""
Model training for DMT2 risk prediction assessment.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import argparse

from args import Training
from data.PMBB import PMBBDataset


def train(args: argparse.Namespace) -> None:
    dataset = PMBBDataset(
        filenames=args.datafiles,
        cache_path=args.cache_path,
        verbose=args.verbose,
        seed=args.seed
    )


if __name__ == "__main__":
    train(Training.build_args())
