import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..src.metrics import CNVMetric

TRAIN_FOLDER = "../train"
TEST_FOLDER = "../test"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TRAIN_PATH = "/".join([TRAIN_FOLDER, TRAIN_FILE])
TEST_PATH = "/".join([TEST_FOLDER, TEST_FILE])

columns_to_drop = [
    "chr",
    "start",
    "end",
    "cnv_type",
    "intq",
    "overlap",
    "BAM_CMATCH",
    "STAT_CROSS",
    "STAT_CROSS2",
    "BAM_CROSS",
    "PR_5",
    "NXT_5",
    "PR_10",
    "NXT_10",
    "PR_20",
    "NXT_20",
]


def _load_train_files(
    lbl_e: LabelEncoder, columns_to_drop: list
) -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(TRAIN_PATH, sep=",")
    train["cnv_type"] = lbl_e.fit_transform(train["cnv_type"])
    X = train.drop(columns_to_drop, axis=1)
    y = train["cnv_type"]
    return X, y


def _load_test_files(
    lbl_e: LabelEncoder, columns_to_drop: list
) -> tuple[pd.DataFrame, pd.Series]:
    df_test = pd.read_csv(TEST_PATH, sep=",")
    df_test["cnv_type"] = lbl_e.transform(df_test["cnv_type"])
    X_test_sim = df_test.drop(columns_to_drop, axis=1)
    y_test_sim = df_test["cnv_type"]
    return X_test_sim, y_test_sim


def get_CNV_metric_results(y_preds: np.ndarray, lbl_e: LabelEncoder) -> dict:
    pred_str = lbl_e.inverse_transform(y_preds)
    df = pd.read_csv(TEST_PATH, sep=",")
    df["pred"] = pred_str
    CNV_metric = CNVMetric(df)
    return CNV_metric


if __name__ == "__main__":
    lbl_e = LabelEncoder()
    scaler = StandardScaler()
    X_train, y_train = _load_train_files(lbl_e, columns_to_drop)
    X_test, y_test = _load_test_files(lbl_e, columns_to_drop)
    X_train = scaler.fit_transform(X_train)
    clf = LogisticRegression(random_state=0, max_iter=5000, verbose=2).fit(
        X_train, y_train
    )
    X_test = scaler.transform(X_test)
    y_preds = clf.predict(X_test)
    y_true = y_test

    metrics = get_CNV_metric_results(y_preds, lbl_e)
    cnv_metrics = metrics.get_metrics()
    cnv_metric = metrics.base_metric(w=2)

    with open("../results/simple_log_reg.txt", "w") as f:
        f.write("Simple logistic regression\n")
        f.write(f"CNV base metric: {cnv_metric}\n")
        f.write(
            f"FBeta score: {fbeta_score(y_test, y_preds, beta=3, average='macro')}\n"
        )
        f.write(f"CNV metrics: {cnv_metrics}\n")
        f.write(classification_report(y_test, y_preds, zero_division=True))
        f.write(str(confusion_matrix(y_test, y_preds)))
