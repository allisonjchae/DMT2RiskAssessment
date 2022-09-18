"""
Utility tools to generate pairplots and estimate KL divergence between
dataset distributions.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
import pandas as pd
from typing import Sequence, Union

from utils import plot_config


def covar_plot(
    feature1: str,
    feature2: str,
    datasets: Sequence[pd.DataFrame],
    labels: Sequence[str],
    colors: Sequence[str] = ["#008D1C", "#009ADE", "#F28522"],
    markers: Sequence[str] = ["s", "o", "v"],
    alphas: Sequence[float] = [0.15, 0.55, 0.2],
    savepath: Union[Path, str] = None
) -> None:
    """
    Generates pairplot for the specified features feature1 and feature2.
    Input:
        feature1: feature on the x axis.
        feature2: feature on the y axis.
        datasets: datasets to plot.
        labels: labels for the datasets.
        colors: plot colors for the datasets.
        markers: plot markers for the datasets.
        alphas: plot transparencies for the datasets.
        savepath: filepath to save the pairplot to.
    Returns:
        None.
    """
    plt.figure(figsize=(6, 10))
    for ds, label, c, m, a in zip(datasets, labels, colors, markers, alphas):
        f1 = ds[feature1]
        f2 = ds[feature2]
        plt.scatter(f1, f2, label=label, alpha=a, color=c, marker=m)
    plt.legend(loc="upper left")
    labels_to_axes = {
        "RESULT_VALUE_NUM": "Hemoglobin A1c (%)",
        "AGE": "Age (Years)",
        "BMI": r"Body Mass Index (kg/m$^2$)"
    }
    plt.xlabel(labels_to_axes[feature1])
    plt.ylabel(labels_to_axes[feature2])
    if savepath is None:
        plt.show()
    else:
        plt.savefig(
            savepath,
            transparent=True,
            dpi=600,
            bbox_inches="tight"
        )


def KLDivergence(p: pd.DataFrame, q: pd.DataFrame) -> float:
    """
    Computes the Kullback-Leibler Divergence between two multivariate samples.
    Input:
        p: empirical samples from the approximate distribution of shape Nd.
        q: empirical samples from the ground truth distribution of shape Md.
    Returns:
        KL(P|Q).
    """
    # There are 2 genders, 6 races, 10 age categories, 4 BMI categories, and
    # 3 A1C categories. Each datapoint is assigned into one of the categories
    # for each feature type below:
    #
    # A. GENDER (self-reported): [Male] or [Female]
    # B. RACE (self-reported): [White] or [Black] or [Asian] or [Am Ind Ak
    #    Native] or [Hi Pac Island] or [Other/Unknown]
    # C. Age (years): [0-9] or [10-19] or [20-29] or [30-39] or [40-49] or
    #    [50-59] or [60-69] or [70-79] or [80-89] or [90+]
    # D. BMI (kg / m^2): [<30] or [30-35] or [35-40] or [40+]
    # E. HbA1c (% A1C): [<5.7] or [5.7-6.5] or [>6.5]
    p_counts = np.zeros((2, 6, 10, 4, 3))
    race_map = {
        "WHITE": 0,
        "BLACK": 1,
        "ASIAN": 2,
        "AM IND AK NATIVE": 3,
        "HI PAC ISLAND": 4,
    }
    for i in range(p.shape[0]):
        item = p.iloc[i]
        gender_idx = int(item["Sex"].title() == "Female")
        race_idx = 5
        if item["RACE_CODE"] in race_map.keys():
            race_idx = race_map[item["RACE_CODE"].upper()]
        age_idx = item["AGE"] // 10
        bmi_idx = int(item["BMI"] >= 30.0)
        bmi_idx += int(item["BMI"] >= 35.0)
        bmi_idx += int(item["BMI"] >= 40.0)
        a1c_idx = int(item["RESULT_VALUE_NUM"] >= 5.7)
        a1c_idx += int(item["RESULT_VALUE_NUM"] >= 6.5)
        p_counts[gender_idx, race_idx, age_idx, bmi_idx, a1c_idx] += 1
    p_probs = np.divide(p_counts, np.sum(p_counts))

    q_counts = np.zeros((2, 6, 10, 4, 3))
    for j in range(q.shape[0]):
        item = q.iloc[j]
        gender_idx = int(item["Sex"].title() == "Female")
        race_idx = 5
        if item["RACE_CODE"] in race_map.keys():
            race_idx = race_map[item["RACE_CODE"].upper()]
        age_idx = item["AGE"] // 10
        bmi_idx = int(item["BMI"] >= 30.0)
        bmi_idx += int(item["BMI"] >= 35.0)
        bmi_idx += int(item["BMI"] >= 40.0)
        a1c_idx = int(item["RESULT_VALUE_NUM"] >= 5.7)
        a1c_idx += int(item["RESULT_VALUE_NUM"] >= 6.5)
        q_counts[gender_idx, race_idx, age_idx, bmi_idx, a1c_idx] += 1
    q_probs = np.divide(q_counts, np.sum(q_counts))

    return np.nansum(
        np.multiply(
            p_probs,
            np.log(np.divide(p_probs, q_probs + np.finfo(np.float64).eps))
        )
    )


def generate_pair_plots():
    plot_config()
    bmi_scale_factor = 703.0

    data_dir = "./data"
    pmbb_outpatient_fn = os.path.join(data_dir, "seed_42", "test_dataset.pkl")
    pmbb_inpatient_fn = os.path.join(
        data_dir, "seed_42", "ood_dataset_inpatient.pkl"
    )
    iraqi_fn = os.path.join(data_dir, "iraqi_dataset.csv")

    with open(pmbb_outpatient_fn, "rb") as outpatient_f:
        pmbb_outpatient = pickle.load(outpatient_f)
    pmbb_outpatient["BMI"] = bmi_scale_factor * pmbb_outpatient[
        "WEIGHT_LBS"
    ] / (pmbb_outpatient["HEIGHT_INCHES"] * pmbb_outpatient["HEIGHT_INCHES"])
    with open(pmbb_inpatient_fn, "rb") as inpatient_f:
        pmbb_inpatient = pickle.load(inpatient_f)
    pmbb_inpatient["BMI"] = bmi_scale_factor * pmbb_inpatient[
        "WEIGHT_LBS"
    ] / (pmbb_inpatient["HEIGHT_INCHES"] * pmbb_inpatient["HEIGHT_INCHES"])
    iraqi = pd.read_csv(iraqi_fn)

    datasets = [pmbb_inpatient, pmbb_outpatient, iraqi]
    labels = ["PMBB Inpatient", "PMBB Outpatient", "Iraqi Medical Center"]

    # Generate and save covariance plots.
    covar_plot(
        "AGE",
        "RESULT_VALUE_NUM",
        datasets,
        labels,
        savepath="./docs/age_vs_a1c.png"
    )
    covar_plot(
        "AGE",
        "BMI",
        datasets,
        labels,
        savepath="./docs/age_vs_bmi.png"
    )
    covar_plot(
        "BMI",
        "RESULT_VALUE_NUM",
        datasets,
        labels,
        savepath="./docs/bmi_vs_a1c.png"
    )


def compute_kl_divergence():
    bmi_scale_factor = 703.0

    data_dir = "./data"
    pmbb_outpatient_fn = os.path.join(data_dir, "seed_42", "test_dataset.pkl")
    pmbb_inpatient_fn = os.path.join(
        data_dir, "seed_42", "ood_dataset_inpatient.pkl"
    )
    iraqi_fn = os.path.join(data_dir, "iraqi_dataset.csv")

    with open(pmbb_outpatient_fn, "rb") as outpatient_f:
        pmbb_outpatient = pickle.load(outpatient_f)
    pmbb_outpatient["BMI"] = bmi_scale_factor * pmbb_outpatient[
        "WEIGHT_LBS"
    ] / (pmbb_outpatient["HEIGHT_INCHES"] * pmbb_outpatient["HEIGHT_INCHES"])
    pmbb_outpatient = pmbb_outpatient[pmbb_outpatient["BMI"] < 1000]
    with open(pmbb_inpatient_fn, "rb") as inpatient_f:
        pmbb_inpatient = pickle.load(inpatient_f)
    pmbb_inpatient["BMI"] = bmi_scale_factor * pmbb_inpatient[
        "WEIGHT_LBS"
    ] / (pmbb_inpatient["HEIGHT_INCHES"] * pmbb_inpatient["HEIGHT_INCHES"])
    pmbb_inpatient = pmbb_inpatient[pmbb_inpatient["BMI"] < 1000]
    iraqi = pd.read_csv(iraqi_fn)

    with open(os.path.join("./data/seed_42/train_dataset.pkl"), "rb") as f:
        train_dataset = pickle.load(f)
    train_dataset["BMI"] = bmi_scale_factor * train_dataset[
        "WEIGHT_LBS"
    ] / (train_dataset["HEIGHT_INCHES"] * train_dataset["HEIGHT_INCHES"])

    print(
        "KL(Iraqi | PMBB Outpatient):",
        KLDivergence(iraqi, train_dataset)
    )
    print(
        "KL(PMBB Inpatient | PMBB Outpatient):",
        KLDivergence(pmbb_inpatient, train_dataset)
    )
    print(
        "KL(PMBB Outpatient Test | PMBB Outpatient):",
        KLDivergence(pmbb_outpatient, train_dataset)
    )


if __name__ == "__main__":
    # Generate pair plots.
    generate_pair_plots()
    # Compute KL Divergences.
    compute_kl_divergence()
