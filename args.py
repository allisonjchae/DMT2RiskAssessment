"""
CLI-friendly argument parser for DMT2 Risk Assessment Model training.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License.
"""
import argparse
from pathlib import Path


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A1C estimator using image data."
    )

    parser.add_argument(
        "--biomarkers_datapath",
        type=Path,
        required=True,
        help="Folder path to biomarkers data containing A1C CSV dataset."
    )
    parser.add_argument(
        "--steatosis_datapath",
        type=Path,
        required=True,
        help="Path to steatosis image data CSV file."
    )
    parser.add_argument(
        "--visceral_fat_datapath",
        type=Path,
        required=True,
        help="Path to visceral fat image data CSV file."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose outputs."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers. Default 0."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs. Default 50."
    )
    parser.add_argument(
        "--chans",
        type=int,
        default=128,
        help="Number of intermediate channels in MLP."
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Total number of layers in MLP."
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        choices=("Sigmoid", "Softplus", "ReLU", "LeakyReLU"),
        help="Activation function in MLP."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Default is time since epoch."
    )
    parser.add_argument(
        "--log_train_loss_every_n_steps",
        type=int,
        default=100,
        help="Log average training loss over specified number of steps."
    )
    parser.add_argument(
        "--data_split",
        type=int,
        nargs=3,
        default=(80, 10, 10),
        help="Data split percentages into training, validation, and test data."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MLP",
        choices=("MLP", "XGBoost", "MLR"),
        help="Model to use. Default MLP for A1C prediction."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    final_help = "If true, treat as the final model and combine training and "
    final_help += "validation data."
    parser.add_argument("--final", action="store_true", help=final_help)

    return parser.parse_args()
