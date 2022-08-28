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
        parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Batch size. Default 1."
        )
        parser.add_argument(
            "--max_epochs",
            type=int,
            default=100,
            help="Maximum training epochs. Default 100."
        )
        parser.add_argument(
            "--gpu",
            type=int,
            default=None,
            help="Index of the GPU to use. Default use CPU only."
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            help="Optimizer for model training. Default Adam optimizer."
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Learning rate. Default 0.001."
        )
        parser.add_argument(
            "--partitions",
            type=float,
            nargs=3,
            default=[0.8, 0.1, 0.1],
            help="Training, validation, and test set fractional partitions."
        )
        identifier_help = "Specifies key for data points. If `A1C`, each A1C "
        identifier_help += "value is treated as a separate datapoint and "
        identifier_help += "patients can be represented multiple times. "
        identifier_help += "If `PMBB_ID`, only the most recent A1C value for "
        identifier_help += "a patient is used."
        parser.add_argument(
            "--identifier",
            type=str,
            choices=("A1C", "PMBB_ID"),
            default="A1C",
            help=identifier_help,
        )
        parser.add_argument(
            "--pmbb_to_penn_fn",
            type=str,
            default="./data/accession_to_pmbb.csv",
            help="File mapping of PMBB accessions to Penn accessions."
        )
        parser.add_argument(
            "--penn_to_date_fn",
            type=str,
            default="./data/sectra_syngo_merge_3-1-19_WITHCPT.csv",
            help="File mapping of Penn accessions to study dates."
        )
        parser.add_argument(
            "--use_clinical",
            action="store_true",
            help="Use clinically derived features as model input."
        )
        parser.add_argument(
            "--use_idp",
            action="store_true",
            help="Use image derived phenotype (IDP) features as model input."
        )

        return parser.parse_args()
