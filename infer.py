"""
Model inference for DMT2 risk prediction assessment.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import argparse
from functools import partialmethod
import numpy as np
import os
from pytorch_tabular import TabularModel
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional

from args import Inference
from utils import seed_everything


def infer(
    args: argparse.Namespace,
    use_clinical: bool = True,
    use_idp: bool = True,
    save_results: bool = True,
    perturbation_lims: Optional[Dict[str, float]] = None,
    seed_offset: Optional[int] = 0
) -> pd.DataFrame:
    seed_everything(args.seed + seed_offset, use_deterministic=True)

    tabular_model = TabularModel.load_from_checkpoint(args.model_dir)
    inference_data = pd.read_pickle(args.datapath)
    if args.classifier:
        inference_data["RESULT_VALUE_NUM"] = (
            inference_data["RESULT_VALUE_NUM"] >= args.A1C_threshmin
        )

    if perturbation_lims is not None:
        rng = np.random.RandomState(seed=args.seed + seed_offset)
        for name, lim in perturbation_lims.items():
            if name not in inference_data.columns:
                continue
            perturbations = rng.uniform(
                low=-lim, high=lim, size=(inference_data.shape[0]),
            )
            inference_data[name] = inference_data[name] + perturbations
    if args.intelligent:
        if use_clinical:
            bmi_scale_factor = 703.0
            inference_data["BMI"] = bmi_scale_factor * inference_data[
                "WEIGHT_LBS"
            ] / (
                inference_data["HEIGHT_INCHES"] * inference_data[
                    "HEIGHT_INCHES"
                ]
            )
            inference_data.drop(
                labels=["HEIGHT_INCHES", "WEIGHT_LBS"], axis=1, inplace=True
            )
        if use_idp:
            inference_data["HEPATIC_FAT"] = (
                inference_data["LIVER_MEAN_HU"] - inference_data[
                    "SPLEEN_MEAN_HU"
                ]
            )
            inference_data.drop(
                labels=["LIVER_MEAN_HU", "SPLEEN_MEAN_HU"],
                axis=1,
                inplace=True
            )
    inference_results = tabular_model.predict(inference_data)
    if save_results:
        inference_results.to_pickle(os.path.abspath(args.savepath))
        print(f"Saved inference results to {os.path.abspath(args.savepath)}")

    return inference_results


def perturb(
    args: argparse.Namespace,
    num_perturbations: int = 100,
    use_clinical: bool = True,
    use_idp: bool = False,
    normalize_loss: bool = True
) -> float:
    non_perturbed = infer(
        args,
        use_clinical=use_clinical,
        use_idp=use_idp,
        save_results=False,
        perturbation_lims=None
    )
    if normalize_loss:
        features = [
            "AGE",
            "BP_SYSTOLIC",
            "BP_DIASTOLIC",
            "HEIGHT_INCHES",
            "WEIGHT_LBS",
            "BMI",
            "RESULT_VALUE_NUM",
        ]
        std_vals = {}
        for feature in features:
            if feature not in non_perturbed.columns:
                continue
            std_vals[feature] = np.std(non_perturbed[feature], ddof=1)
    non_perturbed_preds = non_perturbed.RESULT_VALUE_NUM_prediction
    perturb_limit = {
        "AGE": 3.0,
        "BP_SYSTOLIC": 5.0,
        "BP_DIASTOLIC": 5.0,
        "HEIGHT_INCHES": 3.0,
        "WEIGHT_LBS": 5.0,
        "BMI": 2.0
    }
    M = np.zeros((non_perturbed_preds.shape[0],), dtype=np.float32)
    for i in range(num_perturbations):
        perturbed = infer(
            args,
            use_clinical=use_clinical,
            use_idp=use_idp,
            save_results=False,
            perturbation_lims=perturb_limit,
            seed_offset=i
        )
        perturbed_preds = perturbed.RESULT_VALUE_NUM_prediction
        mu = np.abs(perturbed_preds - non_perturbed_preds)
        if normalize_loss:
            mu = mu / std_vals["RESULT_VALUE_NUM"]
            perturbation_norm = np.zeros(
                (perturbed_preds.shape[0],), dtype=np.float32
            )
            for name, _ in perturb_limit.items():
                if name not in perturbed.columns:
                    continue
                perturbation_norm += np.square(
                    (perturbed[name] - non_perturbed[name]) / (
                        std_vals[name]
                    )
                )
            mu = np.divide(mu, np.sqrt(perturbation_norm))
        M += mu

    return np.mean(np.divide(M, float(num_perturbations)))


if __name__ == "__main__":
    mode = "infer"
    # mode = "perturb"
    if mode == "infer":
        infer(Inference.build_args())
    elif mode == "perturb":
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        num_perturbations = 100
        use_clinical = True
        use_idp = False
        normalize_loss = True
        print(perturb(
            args=Inference.build_args(),
            num_perturbations=num_perturbations,
            use_clinical=use_clinical,
            use_idp=use_idp,
            normalize_loss=normalize_loss
        ))
