import random

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import LabelEncoder

from ..src.metrics import CNVMetric

TRAIN_FOLDER = "../train"
TEST_FOLDER = "../test"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TRAIN_PATH = "/".join([TRAIN_FOLDER, TRAIN_FILE])
TEST_PATH = "/".join([TEST_FOLDER, TEST_FILE])


def load_test_files() -> tuple[pd.DataFrame, pd.Series]:
    df_test = pd.read_csv(TEST_PATH, sep=",")
    X_test_sim = df_test.drop(columns=["cnv_type"], axis=1)
    y_test_sim = df_test["cnv_type"]
    return X_test_sim, y_test_sim


def count_prob_for_each_class(y_test: np.ndarray) -> dict:
    probs = {}
    for i in range(3):
        probs[i] = (y_test == i).sum() / len(y_test)
    return probs


def get_CNV_metric_results(
    X_test: pd.DataFrame, y_test: np.ndarray, y_preds: np.ndarray, lbl_e: LabelEncoder
) -> dict:
    pred_str = lbl_e.inverse_transform(y_preds)
    y_test_str = lbl_e.inverse_transform(y_test)
    df_preds = pd.DataFrame({"cnv_type": y_test_str, "pred": pred_str})
    df = pd.concat([X_test, df_preds], axis=1)
    CNV_metric = CNVMetric(df)
    return CNV_metric.get_metrics()


if __name__ == "__main__":
    lbl_e = LabelEncoder()
    X_test, y_test = load_test_files()
    y_test = lbl_e.fit_transform(y_test)
    probs = count_prob_for_each_class(y_test)
    y_preds = [
        random.choices([0, 1, 2], weights=list(probs.values()), k=1)[0]
        for _ in range(len(y_test))
    ]
    cnv_metrics = get_CNV_metric_results(X_test, y_test, y_preds, lbl_e)

    with open("../results/simple_heuristic.txt", "w") as f:
        f.write("Simple heuristic\n")
        f.write(
            f"FBeta score: {fbeta_score(y_test, y_preds, beta=3, average='weighted')}\n"
        )
        f.write("CNV metrics:\n")
        for key, value in cnv_metrics.items():
            f.write(f"{key}: {value}\n")
