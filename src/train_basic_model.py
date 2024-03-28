import logging
import os
import pickle
from typing import Union

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier

from analysis.plots import plot_PR

from .metrics import CNVMetric
from .paths import (
    BASIC_MODEL_PATH,
    MODELS_FOLDER,
    OVR_BASIC_MODEL_PATH,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)

logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)


class Train:
    def __init__(self, eda: bool, dtype: dict) -> None:
        self.eda = eda
        self.dtype = dtype
        self.columns_to_drop = [
            "chr",
            "start",
            "end",
            "cnv_type",
            "intq",
            "overlap",
            "BAM_CMATCH",
        ]
        self.results: list = []
        self.best_score: float = 0.0
        self.best_params: dict = {}
        self.mapping = {"del": 0, "dup": 1, "normal": 2}

    def train(self):
        logging.info("Loading train files...")
        X_train, y_train = self._load_train_files()
        logging.info("Loading val files...")
        X_val, y_val = self._load_val_files()
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            timeout=3600 * 8,
        )
        # Get the best hyperparameters
        logging.info("Saving best hyperparameters...")
        results_df = pd.DataFrame(self.results)
        self.best_params = (
            results_df.sort_values(by="f1", ascending=False).iloc[0].to_dict()
        )
        self.__save_HP_results(results_df)
        X = pd.concat([X_train, X_val])
        y = pd.concat([y_train, y_val])

        X = self._preprocess_data(X)

        if self.best_params["scaler"] == "log":
            self.log = True
            X, _ = self.__log_transform(X, X)
        elif self.best_params["scaler"] == "standard":
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            pass

        if self.best_params["model"] == "XGBoost":
            best_model = self.__get_xgboost_model(
                self.best_params["class_weight"],
                self.best_params["params"],
            )
        elif self.best_params["model"] == "LightGBM":
            best_model = self.__get_lightgbm_model(
                self.best_params["class_weight"],
                self.best_params["params"],
            )
        elif self.best_params["model"] == "RandomForest":
            best_model = self.__get_random_forest_model(
                self.best_params["class_weight"],
                self.best_params["params"],
            )
        logging.info("Training best model...")
        best_model.fit(X, y)
        os.makedirs(MODELS_FOLDER, exist_ok=True)
        if self.eda:
            logging.info("Training OneVsRest model...")
            y_bin = label_binarize(y, classes=[*range(3)])
            print(np.unique(y_bin))
            one_vs_rest_model = OneVsRestClassifier(best_model)
            one_vs_rest_model.fit(X, y_bin)
            with open(OVR_BASIC_MODEL_PATH, "wb") as f:
                pickle.dump(one_vs_rest_model, f)
        with open(BASIC_MODEL_PATH, "wb") as f:
            pickle.dump(best_model, f)

    def evaluate_on_test_data(self):
        logging.info("Evaluating on test data...")
        X_test, y_test = self._load_test_files()
        if not self.best_params:
            self.best_params = (
                pd.read_csv("HP_results/ML_best_HP.csv")
                .sort_values("f1", ascending=False)
                .iloc[0]
                .to_dict()
            )
        X_test = self._preprocess_data(X_test)

        if self.best_params["scaler"] == "log":
            X_test, _ = self.__log_transform(X_test, X_test)
        elif self.best_params["scaler"] == "standard":
            X_test = self.scaler.transform(X_test)
        else:
            pass

        model = pickle.load(open(BASIC_MODEL_PATH, "rb"))
        pred = model.predict(X_test)
        pred_str = self.map_function(pred, reverse_mapping=True)
        df = pd.read_csv(TEST_PATH, sep=",")
        df["pred"] = pred_str
        metrics = CNVMetric(df)
        metrics_res = metrics.get_metrics()
        if self.eda:
            self._get_precision_recall_curve(X_test, y_test)
        with open("results/ML_model.txt", "w") as f:
            print("ML MODEL", file=f)
            print(f"Params: {self.best_params}", file=f)
            print(
                f"F1 SCORE: {f1_score(y_test, pred, average='macro')}",
                file=f,
            )
            print(classification_report(y_test, pred, zero_division=True), file=f)
            print(confusion_matrix(y_test, pred), file=f)
            print(metrics_res, file=f)

    # Define the objective function to optimize
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        # Define the hyperparameters to search over
        model_type = trial.suggest_categorical(
            "model_type", ["LightGBM", "XGBoost", "RandomForest", "LogisticRegression"]
        )
        class_weight = trial.suggest_categorical(
            "class_weight",
            [
                None,
                "balanced",
                {0: 3, 1: 3, 2: 1},
                {0: 4, 1: 4, 2: 1},
                {0: 3, 1: 5, 2: 1},
            ],
        )
        scaler = trial.suggest_categorical("scaler", ["log", "standard", None])
        stats1 = trial.suggest_categorical("stats1", [True, False])
        stats2 = trial.suggest_categorical("stats2", [True, False])
        bam_fc = trial.suggest_categorical("bam_fc", [True, False])
        prev_and_next = trial.suggest_categorical("prev_and_next", [True, False])

        if not stats1:
            X_train = X_train.drop(["STAT_CROSS"], axis=1)
            X_val = X_val.drop(["STAT_CROSS"], axis=1)

        if not stats2:
            X_train = X_train.drop(["STAT_CROSS2"], axis=1)
            X_val = X_val.drop(["STAT_CROSS2"], axis=1)

        if not bam_fc:
            X_train = X_train.drop(["BAM_CROSS"], axis=1)
            X_val = X_val.drop(["BAM_CROSS"], axis=1)

        if not prev_and_next:
            X_train = X_train.drop(
                ["PR_5", "NXT_5", "PR_10", "NXT_10", "PR_20", "NXT_20"], axis=1
            )
            X_val = X_val.drop(
                ["PR_5", "NXT_5", "PR_10", "NXT_10", "PR_20", "NXT_20"], axis=1
            )

        if scaler == "log":
            X_train, X_val_res = self.__log_transform(X_train, X_val)
        elif scaler == "standard":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val_res = scaler.transform(X_val)
        else:
            X_val_res = X_val

        if model_type == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=20),
                "max_depth": trial.suggest_int("max_depth", 20, 160, step=20),
                "min_samples_split": trial.suggest_int("min_samples_split", 3, 150),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 60),
            }
            model = self.__get_random_forest_model(class_weight, params)
        elif model_type == "LogisticRegression":
            params = {
                "solver": trial.suggest_categorical(
                    "solver", ["newton-cg", "lbfgs", "sag", "saga"]
                ),
                "C": trial.suggest_float("C", 1e-3, 1.0, log=True),
            }
            model = self.__get_logistic_regression_model(class_weight, params)
        elif model_type == "LightGBM":
            params = {
                "max_depth": trial.suggest_int("max_depth", 20, 160, step=20),
                "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]
                ),
                "num_leaves": trial.suggest_int("num_leaves", 50, 700),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 5000),
                "verbosity": -1,
            }
            model = self.__get_lightgbm_model(class_weight, params)
        elif model_type == "XGBoost":
            params = {
                "verbosity": 0,
                "max_depth": trial.suggest_int("max_depth", 20, 120, step=20),
                "n_estimators": trial.suggest_int("n_estimators", 10, 80, step=10),
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-3, 1.0, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                ),
            }

            if params["booster"] == "dart":
                params["sample_type"] = trial.suggest_categorical(
                    "sample_type", ["uniform", "weighted"]
                )
                params["normalize_type"] = trial.suggest_categorical(
                    "normalize_type", ["tree", "forest"]
                )
                params["rate_drop"] = trial.suggest_float(
                    "rate_drop", 1e-8, 0.4, log=True
                )
                params["skip_drop"] = trial.suggest_float(
                    "skip_drop", 1e-8, 0.4, log=True
                )
            model = self.__get_xgboost_model(class_weight, params)

        # Trenowanie modelu
        model.fit(X_train, y_train)

        # Przewidywanie na zbiorze testowym

        y_pred = model.predict(X_val_res)
        f1 = f1_score(y_val, y_pred, average="macro")

        self.results.append(
            {
                "model": model_type,
                "class_weight": class_weight,
                "scaler": scaler,
                "stats1": stats1,
                "stats2": stats2,
                "bam_fc": bam_fc,
                "prev_and_next": prev_and_next,
                "params": params,
                "f1": f1,
            }
        )
        return f1

    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.best_params["stats1"]:
            X = X.drop(["STAT_CROSS"], axis=1)

        if not self.best_params["stats2"]:
            X = X.drop(["STAT_CROSS2"], axis=1)

        if not self.best_params["bam_fc"]:
            X = X.drop(["BAM_CROSS"], axis=1)

        if not self.best_params["prev_and_next"]:
            X = X.drop(["PR_5", "NXT_5", "PR_10", "NXT_10", "PR_20", "NXT_20"], axis=1)
        return X

    def __get_random_forest_model(
        self,
        class_weight: Union[str, dict, None],
        params: dict,
    ) -> RandomForestClassifier:
        model = RandomForestClassifier(
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        return model

    def __get_xgboost_model(
        self,
        class_weight: Union[str, dict, None],
        params: dict,
    ) -> XGBClassifier:
        model = XGBClassifier(
            scale_pos_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            num_class=3,
            **params,
        )
        return model

    def __get_lightgbm_model(
        self,
        class_weight: Union[str, dict, None],
        params: dict,
    ) -> LGBMClassifier:
        model = LGBMClassifier(
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        return model

    def __get_logistic_regression_model(
        self,
        class_weight: Union[str, dict, None],
        params: dict,
    ) -> LogisticRegression:
        model = LogisticRegression(
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        return model

    def _load_train_files(self) -> tuple[pd.DataFrame, pd.Series]:
        train = pd.read_csv(TRAIN_PATH, dtype=self.dtype, sep=",")
        if self.eda:
            self._perform_eda(train)
        train["cnv_type"] = self.map_function(train["cnv_type"])
        X = train.drop(self.columns_to_drop, axis=1)
        y = train["cnv_type"]
        return X, y

    def _load_test_files(self) -> tuple[pd.DataFrame, pd.Series]:
        df_test = pd.read_csv(TEST_PATH, dtype=self.dtype, sep=",")
        df_test["cnv_type"] = self.map_function(df_test["cnv_type"])
        X_test_sim = df_test.drop(self.columns_to_drop, axis=1)
        y_test_sim = df_test["cnv_type"]
        return X_test_sim, y_test_sim

    def _load_val_files(self) -> tuple[pd.DataFrame, pd.Series]:
        val = pd.read_csv(VAL_PATH, dtype=self.dtype, sep=",")
        val["cnv_type"] = self.map_function(val["cnv_type"])
        X = val.drop(self.columns_to_drop, axis=1)
        y = val["cnv_type"]
        return X, y

    def __log_transform(
        self, X_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train = np.log1p(X_train)
        x_test = np.log1p(x_test)
        return X_train, x_test

    def _perform_eda(self, df: pd.DataFrame):
        with open("plots/EDA_train.txt", "w") as f:
            print(df.groupby("cnv_type")["means"].describe(), file=f)
            print(df.groupby("cnv_type")["std"].describe(), file=f)
            print(df.groupby("cnv_type")["intq"].describe(), file=f)
            print(df.sort_values(by="means", ascending=False).head(15), file=f)

    def _get_precision_recall_curve(self, X_test: pd.DataFrame, y_test: np.ndarray):
        model = pickle.load(open(OVR_BASIC_MODEL_PATH, "rb"))
        y_score = model.predict(X_test)
        y_true = label_binarize(y_test, classes=[*range(3)])
        plot_PR(y_true, y_score)

    def map_function(self, target_values: list, reverse_mapping: bool = False) -> list:
        if reverse_mapping:
            rev_map = {v: k for k, v in self.mapping.items()}
            return list(map(rev_map.get, target_values))
        else:
            return list(map(self.mapping.get, target_values))

    def __save_HP_results(self, results_df: pd.DataFrame):
        os.makedirs("HP_results", exist_ok=True)
        results_sorted = results_df.sort_values(by="f1", ascending=False)
        results_sorted.to_csv("HP_results/ML_best_HP.csv", index=False)
