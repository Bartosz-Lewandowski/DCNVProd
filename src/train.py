import os
import pickle

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from plots import plot_count


class Train:
    def __init__(self, eda: bool) -> None:
        self.eda = eda
        self.create_train_test_folders()

    def prepare_sim(self) -> None:
        sim_data = pd.read_csv("stats/sim/combined.csv", sep=",", dtype={"chr": object})
        test = sim_data[sim_data["chr"] == "13"]
        train = sim_data[sim_data["chr"] != "13"]
        if self.eda:
            plot_count("cnv_type", train, "train_sim")
            plot_count("cnv_type", test, "test_sim")
        train.to_csv("train/sim.csv", index=False)
        test.to_csv("test/sim.csv", index=False)

    def load_train_files(self):
        sim = pd.read_csv("train/sim.csv", sep=",", dtype={"chr": object})
        return sim

    def load_test_sim_files(self, lbl_e):
        df_test_sim = pd.read_csv("test/sim.csv", sep=",", dtype={"chr": object})
        df_test_sim["cnv_type"] = lbl_e.transform(df_test_sim["cnv_type"])
        X_test_sim = df_test_sim.drop(["chr", "start", "end", "cnv_type"], axis=1)
        y_test_sim = df_test_sim["cnv_type"]
        return X_test_sim, y_test_sim

    def train(self):
        df = self.load_train_files()
        lbl_e = LabelEncoder()
        df["cnv_type"] = lbl_e.fit_transform(df["cnv_type"])
        X = df.drop(["chr", "start", "end", "cnv_type"], axis=1)
        y = df["cnv_type"]
        if self.eda:
            self.perform_eda()
        X_res, y_res, res_count = self.undersample(X, y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)
        class_weights = {
            0: 1 / res_count[0],
            1: 1 / res_count[1],
            2: 1 / res_count[2],
        }
        X_test_sim, y_test_sim = self.load_test_sim_files(lbl_e)
        model = self.fit_lgb(X_scaled, y_res, class_weights)
        self.evaluate(model, X_test_sim, y_test_sim, scaler, "SIM")
        os.makedirs("model", exist_ok=True)
        pickle.dump(model, "model/best_model")

    def fit_lgb(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        class_weights: dict,
        sample_weights: pd.Series,
    ) -> LGBMClassifier:
        model = LGBMClassifier(
            seed=42, class_weight=class_weights, n_jobs=10, n_estimators=175
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model

    def undersample(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
        from collections import Counter

        cont = Counter(y_train)
        classes_resampling = {2: int(cont[2] * 0.3)}
        under = RandomUnderSampler(
            sampling_strategy=classes_resampling, random_state=42
        )
        X_res, y_res = under.fit_resample(X_train, y_train)
        res_count = Counter(y_res)
        return X_res, y_res, res_count

    def evaluate(self, model, X_test, y_test, scaler, type):
        pred = model.predict(scaler.transform(X_test))
        with open("metrics.txt", "a") as f:
            print(type, file=f)
            print(
                f"FBETA SCORE: {fbeta_score(y_test, pred, beta = 3, average='macro')}",
                file=f,
            )
            print(classification_report(y_test, pred, zero_division=True), file=f)
            print(confusion_matrix(y_test, pred), file=f)

    def perform_eda(self, df: pd.DataFrame):
        with open("plots/EDA_train.txt", "w") as f:
            print(df.groupby("cnv_type")["means"].describe(), file=f)
            print(df.groupby("cnv_type")["std"].describe(), file=f)
            print(df.groupby("cnv_type")["BAM_CMATCH"].describe(), file=f)
            print(df.sort_values(by="means", ascending=False).head(15), file=f)

    def create_train_test_folders(self):
        os.makedirs("train", exist_ok=True)
        os.makedirs("test", exist_ok=True)
