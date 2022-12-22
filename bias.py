"""
Assessment for model algorithmic bias in DMT2 risk prediction assessment.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pathlib import Path
import scipy.stats as stats
from typing import Union
from utils import plot_config


def bias(datapath: Union[Path, str], verbose: bool = True) -> dict:
    bias_results = {}
    if verbose:
        print(datapath)
    with open(datapath, "rb") as f:
        data = pickle.load(f)
    try:
        target = data.RESULT_VALUE_NUM.to_numpy()
        preds = data.RESULT_VALUE_NUM_prediction.to_numpy()
    except AttributeError:
        preds = np.array(data)
        ref_datapath = os.path.join(
            os.path.dirname(datapath).replace("results", "data"),
            "test_dataset.pkl"
        )
        with open(ref_datapath, "rb") as f:
            data = pickle.load(f)
            target = data.RESULT_VALUE_NUM.to_numpy()
    race = data.RACE_CODE.to_numpy()
    gender = data.Sex.to_numpy()
    try:
        height = data.HEIGHT_INCHES.to_numpy()
        weight = data.WEIGHT_LBS.to_numpy()
        bmi = 703.0 * weight / np.square(height)
    except AttributeError:
        bmi = data.BMI.to_numpy()

    # Analyze by gender.
    female = (preds - target)[gender == "Female"]
    male = (preds - target)[gender == "Male"]
    min_var, max_var = np.var(female), np.var(male)
    if min_var > max_var:
        min_var, max_var = max_var, min_var
    gender = stats.ttest_ind(
        female, male, equal_var=(max_var / min_var <= 4.0)
    )
    female_ci = stats.t.interval(
        confidence=0.95,
        df=(len(female) - 1),
        loc=np.mean(female),
        scale=stats.sem(female)
    )
    male_ci = stats.t.interval(
        confidence=0.95,
        df=(len(male) - 1),
        loc=np.mean(male),
        scale=stats.sem(male)
    )
    bias_results["gender"] = {
        "labels": ["Female", "Male"],
        "cis": [female_ci, male_ci],
        "pvalue": gender.pvalue
    }
    if verbose:
        print("GENDER")
        print(
            "\tFemale:",
            round(np.mean(female), 2),
            u"\u00B1",
            round(np.std(female), 2),
            tuple([round(x, 3) for x in female_ci])
        )
        print(
            "\tMale:",
            round(np.mean(male), 2),
            u"\u00B1",
            round(np.std(male), 2),
            tuple([round(x, 3) for x in male_ci])
        )
        print(f"\tTwo Sample T-test: p = {gender.pvalue:.3f}")

    # Analyze by BMI.
    bmi_thresholds = [0, 30, 35, 40, 1e12]
    bmi_labels = [
        "Not Overweight or Obese", "Overweight", "Obese", "Extremely Obese"
    ]
    bmi_stratified = []
    for i in range(len(bmi_thresholds) - 1):
        bmi_stratified.append(
            (preds - target)[
                (bmi_thresholds[i] <= bmi) & (bmi < bmi_thresholds[i + 1])
            ]
        )
    bmi = stats.f_oneway(*bmi_stratified)
    bmi_cis = [
        stats.t.interval(
            0.95, df=(len(x) - 1), loc=np.mean(x), scale=stats.sem(x)
        )
        for x in bmi_stratified
    ]
    bias_results["bmi"] = {
        "labels": bmi_labels[::-1],
        "cis": bmi_cis[::-1],
        "pvalue": bmi.pvalue
    }
    if verbose:
        print("BMI")
        for label, data, ci in zip(bmi_labels, bmi_stratified, bmi_cis):
            print(
                "\t" + label + ":",
                round(np.mean(data), 2),
                u"\u00B1",
                round(np.std(data), 2),
                tuple([round(x, 3) for x in ci])
            )
        print(f"\tANOVA Test: p = {bmi.pvalue:.3f}")

    # Analyze by self-reported ethnicity.
    ethnicities = list(set(race.tolist()))
    race_stratified = [(preds - target)[race == x] for x in ethnicities]
    ethnicities = [
        ethnicities[i]
        for i in range(len(ethnicities)) if race_stratified[i].shape[0] > 1
    ]
    race_stratified = [x for x in race_stratified if x.shape[0] > 1]
    race = stats.f_oneway(*race_stratified)
    race_cis = [
        stats.t.interval(
            confidence=0.95,
            df=(len(x) - 1),
            loc=np.mean(x),
            scale=stats.sem(x)
        )
        for x in race_stratified
    ]
    bias_results["race"] = {
        "labels": [x.title() for x in ethnicities],
        "cis": race_cis,
        "pvalue": race.pvalue
    }
    if verbose:
        print("RACE")
        for label, data, ci in zip(ethnicities, race_stratified, race_cis):
            print(
                "\t" + label + ":",
                round(np.mean(data), 2),
                u"\u00B1",
                round(np.std(data), 2),
                tuple([round(x, 3) for x in ci])
            )
        print(f"\tANOVA Test: p = {race.pvalue:.3f}")

        print()
    return bias_results


def plot_cis(bias_results: dict, savedir: Union[Path, str] = None) -> None:
    for key in bias_results[list(bias_results.keys())[0]]:
        plt.figure(figsize=(10, 14))
        plt.axvline(color="#6f6f6f", linestyle="-.", alpha=0.5)

        colors = [
            "#E87722",
            "#E87722cc",
            "#002855",
            "#002855cc",
            "#007377",
            "#007377cc",
            "#45484B",
            "#45484Bcc",
        ]
        for i, (fn, co) in enumerate(
            zip(list(bias_results.keys())[::-1], colors)
        ):
            results = bias_results[fn]
            gender = results[key]
            counter = str(len(bias_results.keys()) - i)
            for label, ci in zip(gender["labels"], gender["cis"]):
                plt.plot(
                    ci, [str(counter) + " " + label] * 2, "-o", color=co
                )
        plt.xlabel("Synthetic HbA1c - Ground Truth HbA1c (% A1C)")
        plt.xlabel("Synthetic HbA1c - Ground Truth HbA1c (% A1C)")

        if savedir is None or not os.path.isdir(savedir):
            plt.show()
        else:
            plt.savefig(
                os.path.join(savedir, f"bias_{key}.png"),
                transparent=True,
                dpi=600,
                bbox_inches="tight"
            )
        plt.close()


if __name__ == "__main__":
    datadir = "./results/seed_42"
    datapaths = [
        "ols_test_final.pkl",
        "ols_test_final_intelligent.pkl",
        "node_test_final.pkl",
        "node_test_final_intelligent.pkl",
        "fttransformer_test_final.pkl",
        "fttransformer_test_final_intelligent.pkl",
        "test_final.pkl",
        "test_final_intelligent.pkl",
    ]
    bias_results = {
        fn: bias(os.path.join(datadir, fn)) for fn in datapaths
    }
    plot_config()
    plot_cis(bias_results, savedir="./docs")
