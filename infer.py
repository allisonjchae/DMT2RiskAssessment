"""
Model inference for DMT2 risk prediction assessment.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import argparse
import os
from pytorch_tabular import TabularModel
import pandas as pd

from args import Inference
from train import seed_everything


def infer(args: argparse.Namespace) -> None:
    seed_everything(args.seed, use_deterministic=True)

    tabular_model = TabularModel.load_from_checkpoint(args.model_dir)
    tabular_model.predict(pd.read_pickle(args.datapath)).to_pickle(
        os.path.abspath(args.savepath)
    )
    print(f"Saved inference results to {os.path.abspath(args.savepath)}")

    return


if __name__ == "__main__":
    infer(Inference.build_args())
