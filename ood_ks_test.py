"""
Assessment for out-of-distribution shift via Kolmogorov-Smirnov Test.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import scipy.stats
from typing import Sequence, Union


def pairwise_ks_test(
    datapaths: Sequence[Union[Path, str]], labels: Sequence[str]
) -> None:
    if len(datapaths) != len(labels):
        raise ValueError("Number of datapaths and labels must match.")

    data = []
    for fn in datapaths:
        if fn.endswith(".csv"):
            data.append(pd.read_csv(fn))
            continue
        with open(fn, "rb") as f:
            data.append(pickle.load(f))
    keys = ["AGE", "RESULT_VALUE_NUM", "BMI"]
    height_key = "HEIGHT_INCHES"
    weight_key = "WEIGHT_LBS"

    for n, k in enumerate(keys):
        print(k)
        for i in range(len(data)):
            x_label = labels[i]
            try:
                x = data[i][k]
            except KeyError:
                x_height, x_weight = data[i][height_key], data[i][weight_key]
                x = 703.0 * x_weight / np.square(x_height)
            for j in range(i, len(data)):
                y_label = labels[j]
                try:
                    y = data[j][k]
                except KeyError:
                    y_height = data[j][height_key]
                    y_weight = data[j][weight_key]
                    y = 703.0 * y_weight / np.square(y_height)
                if x_label == y_label:
                    continue
                p_value = scipy.stats.kstest(x, y).pvalue
                if p_value < 0.01:
                    p_value = f"{p_value:.3E}"
                else:
                    p_value = str(round(p_value, 3))
                print(f"\t{x_label} + {y_label}:", p_value)
        if n < len(keys) - 1:
            print()


if __name__ == "__main__":
    labels = ["PMBB Inpatient", "PMBB Outpatient", "Iraqi Medical Center"]
    datadir = os.path.join("./", "data")
    seed = 42
    datapaths = [
        os.path.join(datadir, f"seed_{seed}", "ood_dataset_inpatient.pkl"),
        os.path.join(datadir, f"seed_{seed}", "test_dataset.pkl"),
        os.path.join(datadir, "iraqi_dataset.csv")
    ]
    pairwise_ks_test(datapaths, labels)
