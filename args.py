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
            "--model",
            type=str,
            choices=(
                "FTTransformer",
                "AutoInt",
                "NODEModel",
                "TabNet",
                "FCNN",
                "XGBoost",
                "OLS",
                "WRC",
            ),
            required=True,
            help="Tabular model specification."
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
            "--lr_step_size",
            type=int,
            default=25,
            help="Learning rate step size. Default 25."
        )
        parser.add_argument(
            "--lr_gamma",
            type=float,
            default=0.5,
            help="Learning rate decay factor. Default 0.5."
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
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.0,
            help="Dropout probability."
        )
        parser.add_argument(
            "--A1C_range",
            type=float,
            nargs=2,
            default=(3.0, 15.0),
            help="Range of possible A1C values. Default 3.0 to 15.0 percent."
        )
        parser.add_argument(
            "--intelligent",
            action="store_true",
            help="Use intelligently derived clinical variables."
        )
        parser.add_argument(
            "--classifier",
            action="store_true",
            help="Train a binary classifier instead of an SynthA1c predictor."
        )
        parser.add_argument(
            "--A1C_threshmin",
            type=float,
            choices=(5.7, 6.5),
            default=6.5,
            help="A1C threshold for classification."
        )

        return parser.parse_args()


class Inference:
    def build_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="DMT2 risk prediction model inference."
        )

        parser.add_argument(
            "--datapath",
            type=str,
            required=True,
            help="Path to pickled dataframe to run inference on."
        )
        parser.add_argument(
            "--model_dir",
            type=str,
            required=True,
            help="Path to model files for inference."
        )
        parser.add_argument(
            "--savepath",
            type=str,
            required=True,
            help="Filepath to save inference results to."
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Optional random seed. Default 42."
        )
        parser.add_argument(
            "--classifier",
            action="store_true",
            help="Specify binary classifier model."
        )
        parser.add_argument(
            "--A1C_threshmin",
            type=float,
            choices=(5.7, 6.5),
            default=6.5,
            help="A1C threshold for classification tasks."
        )
        parser.add_argument(
            "--intelligent",
            action="store_true",
            help="Use intelligently derived clinical variables."
        )

        return parser.parse_args()
