"""
Model inference using the risk test offered by the American Diabetes
Association (ADA) and the Centers for Disease Control and Prevention (CDC).

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.

Reference Link: https://diabetes.org/diabetes/risk-test
"""
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


def ada_risk(dataset: pd.DataFrame) -> np.ndarray:
    points = np.zeros(dataset.shape[0])
    # 1. How old are you?
    points += dataset["AGE"].to_numpy() >= 40
    points += dataset["AGE"].to_numpy() >= 50
    points += dataset["AGE"].to_numpy() >= 60

    # 2. Are you a main or a woman?
    points += dataset["Sex"].to_numpy() == "Male"

    # 3. If you are a woman, have you ever been diagnosed with gestational
    # diabetes?
    # Cannot implement since we do not have this data.

    # 4. Do you have a mother, father, sister, or brother with diabetes?
    # Cannot implement since we do not have this data.

    # 5. Have you ever been diagnosed with high blood pressure?
    points += np.logical_or(
        dataset["BP_SYSTOLIC"].to_numpy() >= 130,
        dataset["BP_DIASTOLIC"].to_numpy() >= 80
    )

    # 6. Are you physically active?
    # Cannot implement since we do not have this data.

    # 7. What is your weight category?
    hw_scores = []
    hw_chart = {
        58: [119, 143, 191],
        59: [124, 148, 198],
        60: [128, 153, 204],
        61: [132, 158, 211],
        62: [136, 164, 218],
        63: [141, 169, 225],
        64: [145, 174, 232],
        65: [150, 180, 240],
        66: [155, 186, 247],
        67: [159, 191, 255],
        68: [164, 197, 262],
        69: [169, 203, 270],
        70: [174, 209, 278],
        71: [179, 215, 286],
        72: [184, 221, 294],
        73: [189, 227, 302],
        74: [194, 233, 311],
        75: [200, 240, 319],
        76: [205, 246, 328],
    }
    for h, w, race in zip(
        dataset["HEIGHT_INCHES"], dataset["WEIGHT_LBS"], dataset["RACE_CODE"]
    ):
        key = max(min(int(h), max(hw_chart.keys())), min(hw_chart.keys()))
        hw_points = 0
        for i, cutoff in enumerate(hw_chart[key]):
            if i == 0 and race.upper() == "ASIAN":
                hw_points += int(w >= (cutoff - 15))
            else:
                hw_points += int(w >= cutoff)
        hw_scores.append(hw_points)
    points += np.array(hw_scores)

    return points


if __name__ == "__main__":
    data_fn = os.path.join("./data/seed_42", "test_dataset.pkl")
    classifier = "diabetes"  # One of [`diabetes`, `prediabetes`].

    a1c_threshold = {
        "diabetes": 6.5,
        "prediabetes": 5.7
    }
    with open(data_fn, "rb") as f:
        data = pickle.load(f)
    scores = ada_risk(data)
    t2dm_status = (
        data["RESULT_VALUE_NUM"].to_numpy() >= a1c_threshold[classifier]
    )

    for score_thresh in range(int(min(scores)), int(max(scores)) + 1):
        acc = accuracy_score(t2dm_status, scores >= score_thresh)
        print(score_thresh, acc)

    cutoff = {
        "diabetes": 5,
        "prediabetes": 3,
    }
    tp, fp, tn, fn = 0, 0, 0, 0
    for ada_score, gt in zip(scores, t2dm_status):
        if gt:
            if ada_score >= cutoff[classifier]:
                tp += 1
            else:
                fn += 1
        else:
            if ada_score >= cutoff[classifier]:
                fp += 1
            else:
                tn += 1
    eps = np.finfo(np.float64).eps
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    print(f"N = {len(scores)}")
    print(f"Sensitivity (Recall): {recall:.3f}")
    print(f"Precision (PPV): {precision:.3f}")
    print(f"Specificity: {(tn / (tn + fp + eps)):.3f}")
