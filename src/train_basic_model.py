import os
import pickle

import numpy as np
import optuna
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import (
    BASIC_MODEL_FOLDER,
    BASIC_MODEL_PATH,
    FEATURES_COMBINED_FILE,
    STATS_FOLDER,
    TEST_FOLDER,
    TEST_PATH,
    TRAIN_FOLDER,
    TRAIN_PATH,
)
from .metrics import CNVMetric


class Train:
    def __init__(self, eda: bool) -> None:
        self.eda = eda
        self.data_file = "/".join([STATS_FOLDER, FEATURES_COMBINED_FILE])
        self._create_train_test_folders()
        self.lbl_e = LabelEncoder()
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
        self.stats1 = True
        self.stats2 = True
        self.bam_fc = True
        self.scaler = False
        self.log = False

    def prepare_data(self) -> None:
        sim_data = pd.read_csv(
            self.data_file,
            sep=",",
            dtype={
                "chr": "int8",
                "start": "int32",
                "end": "int32",
                "overlap": "float32",
                "intq": "float16",
                "means": "float16",
                "std": "float16",
                "BAM_CMATCH": "int32",
                "BAM_CINS": "int16",
                "BAM_CDEL": "int16",
                "BAM_CSOFT_CLIP": "int16",
                "NM tag": "int16",
                "STAT_CROSS": "float16",
                "STAT_CROSS2": "float16",
                "BAM_CROSS": "int64",
            },
        )
        test = sim_data[sim_data["chr"].isin([13, 7])]
        train = sim_data[sim_data["chr"].isin([1, 2, 3, 9])]
        train.to_csv(TRAIN_PATH, index=False)
        test.to_csv(TEST_PATH, index=False)

    def train(self):
        X, y = self._load_train_files()
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective(trial, X, y), timeout=3600 * 12)

        # Get the best hyperparameters
        results_df = pd.DataFrame(self.results)
        best_params = results_df.sort_values(by="fbeta", ascending=False).iloc[0]

        if not best_params["stats1"]:
            self.stats1 = False
            X = X.drop(["STAT_CROSS"], axis=1)

        if not best_params["stats2"]:
            self.stats2 = False
            X = X.drop(["STAT_CROSS2"], axis=1)

        if not best_params["bam_fc"]:
            self.bam_fc = False
            X = X.drop(["BAM_CROSS"], axis=1)

        if best_params["undersampling"]:
            X, y = self.__undersample(X, y)
        if best_params["scaler"] == "StandardScaler":
            self.scaler = True
            X, _ = self.__scale_standard_scaler(X, X)
        elif best_params["scaler"] == "log":
            self.log = True
            X, _ = self.__log_transform(X, X)

        if best_params["model"] == "XGBoost":
            best_model = self.__get_xgboost_model(
                best_params["max_depth"],
                best_params["class_weight"],
                best_params["n_estimators"],
                best_params["params"],
            )
        elif best_params["model"] == "LightGBM":
            best_model = self.__get_lightgbm_model(
                best_params["max_depth"],
                best_params["class_weight"],
                best_params["n_estimators"],
                best_params["params"],
            )
        elif best_params["model"] == "RandomForest":
            best_model = self.__get_random_forest_model(
                best_params["max_depth"],
                best_params["class_weight"],
                best_params["n_estimators"],
                best_params["params"],
            )

        best_model.fit(X, y)
        os.makedirs(BASIC_MODEL_FOLDER, exist_ok=True)
        with open(BASIC_MODEL_PATH, "wb") as f:
            pickle.dump(best_model, f)

    def evaluate_on_test_data(self):
        X_test, y_test = self._load_test_files()
        if not self.stats1:
            X_test = X_test.drop(["STAT_CROSS"], axis=1)

        if not self.stats2:
            X_test = X_test.drop(["STAT_CROSS2"], axis=1)

        if not self.bam_fc:
            X_test = X_test.drop(["BAM_CROSS"], axis=1)

        if self.scaler:
            X_test, _ = self.__scale_standard_scaler(X_test, X_test)

        if self.log:
            X_test, _ = self.__log_transform(X_test, X_test)

        model = pickle.load(open(BASIC_MODEL_PATH, "rb"))
        pred = model.predict(X_test)
        pred_str = self.lbl_e.inverse_transform(pred)
        df = pd.read_csv(TEST_PATH, sep=",")
        df["pred"] = pred_str
        metrics = CNVMetric(df)
        metrics_res = metrics.get_metrics()
        with open("metrics_feature_crossing.txt", "a") as f:
            print(
                f"FBETA SCORE: {fbeta_score(y_test, pred, beta = 3, average='macro')}",
                file=f,
            )
            print(classification_report(y_test, pred, zero_division=True), file=f)
            print(confusion_matrix(y_test, pred), file=f)
            print(metrics_res, file=f)

    # Define the objective function to optimize
    def _objective(self, trial, X, y):
        # Define the hyperparameters to search over
        model_type = trial.suggest_categorical(
            "model_type", ["LightGBM", "XGBoost", "RandomForest"]
        )
        max_depth = trial.suggest_int("max_depth", 20, 200, step=20)
        class_weight = trial.suggest_categorical(
            "class_weight", [None, "balanced", {0: 3, 1: 3, 2: 1}]
        )
        scaler = trial.suggest_categorical("scaler", ["StandardScaler", "log"])
        undersampling = trial.suggest_categorical("undersampling", [True, False])
        stats1 = trial.suggest_categorical("stats1", [True, False])
        stats2 = trial.suggest_categorical("stats2", [True, False])
        bam_fc = trial.suggest_categorical("bam_fc", [True, False])

        avg_fbeta = 0
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if not stats1:
                X_train = X_train.drop(["STAT_CROSS"], axis=1)
                X_test = X_test.drop(["STAT_CROSS"], axis=1)

            if not stats2:
                X_train = X_train.drop(["STAT_CROSS2"], axis=1)
                X_test = X_test.drop(["STAT_CROSS2"], axis=1)

            if not bam_fc:
                X_train = X_train.drop(["BAM_CROSS"], axis=1)
                X_test = X_test.drop(["BAM_CROSS"], axis=1)

            # Preprocess the data based on hyperparameters
            if undersampling:
                X_res, y_res = self.__undersample(X_train, y_train)
            else:
                X_res, y_res = X_train, y_train

            if scaler == "StandardScaler":
                X_res, x_test_res = self.__scale_standard_scaler(X_res, X_test)
            elif scaler == "log":
                X_res, x_test_res = self.__log_transform(X_res, X_test)
            else:
                x_test_res = X_test

            if model_type == "RandomForest":
                n_estimators = trial.suggest_int("n_estimators", 20, 300, step=20)
                params = {
                    "min_samples_split": trial.suggest_int("min_samples_split", 3, 150),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 60),
                }
                model = self.__get_random_forest_model(
                    max_depth, class_weight, n_estimators, params
                )
            elif model_type == "LightGBM":
                n_estimators = trial.suggest_int("n_estimators", 20, 300, step=20)
                params = {
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0),
                    "learning_rate": trial.suggest_categorical(
                        "learning_rate", [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]
                    ),
                    "num_leaves": trial.suggest_int("num_leaves", 50, 700),
                    "min_child_samples": trial.suggest_int(
                        "min_child_samples", 20, 5000
                    ),
                    "verbosity": -1,
                }
                model = self.__get_lightgbm_model(
                    max_depth, class_weight, n_estimators, params
                )
            elif model_type == "XGBoost":
                n_estimators = trial.suggest_int("n_estimators", 20, 140, step=20)
                params = {
                    "verbosity": 0,
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
                model = self.__get_xgboost_model(
                    max_depth, class_weight, n_estimators, params
                )

            # Trenowanie modelu
            model.fit(X_res, y_res)

            # Przewidywanie na zbiorze testowym

            y_pred = model.predict(x_test_res)
            fbeta = fbeta_score(y_test, y_pred, beta=3, average="macro")
            print(fbeta)
            avg_fbeta += fbeta

        self.results.append(
            {
                "model": model_type,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "class_weight": class_weight,
                "scaler": scaler,
                "undersampling": undersampling,
                "stats1": stats1,
                "stats2": stats2,
                "bam_fc": bam_fc,
                "params": params,
                "fbeta": avg_fbeta / 3,
            }
        )
        return avg_fbeta / 3

    def __get_random_forest_model(
        self, max_depth: int, class_weight: str, n_estimators: int, params: dict
    ) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        return model

    def __get_xgboost_model(
        self, max_depth: int, class_weight: str, n_estimators: int, params: dict
    ) -> XGBClassifier:
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_pos_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        return model

    def __get_lightgbm_model(
        self, max_depth: int, class_weight: str, n_estimators: int, params: dict
    ) -> LGBMClassifier:
        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        return model

    def _load_train_files(self) -> tuple[pd.DataFrame, pd.Series]:
        train = pd.read_csv(TRAIN_PATH, sep=",")
        if self.eda:
            self._perform_eda(train)
        train["cnv_type"] = self.lbl_e.fit_transform(train["cnv_type"])
        X = train.drop(self.columns_to_drop, axis=1)
        y = train["cnv_type"]
        return X, y

    def _load_test_files(self) -> tuple[pd.DataFrame, pd.Series]:
        df_test = pd.read_csv(TEST_PATH, sep=",")
        df_test["cnv_type"] = self.lbl_e.transform(df_test["cnv_type"])
        X_test_sim = df_test.drop(self.columns_to_drop, axis=1)
        y_test_sim = df_test["cnv_type"]
        return X_test_sim, y_test_sim

    def __log_transform(
        self, X_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train = np.log1p(X_train)
        x_test = np.log1p(x_test)
        return X_train, x_test

    def __scale_standard_scaler(
        self, X_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        s = StandardScaler()
        X_train = s.fit_transform(X_train)
        x_test = s.transform(x_test)
        return X_train, x_test

    def __undersample(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:

        under = RandomUnderSampler(sampling_strategy="majority", random_state=42)
        X_res, y_res = under.fit_resample(X_train, y_train)
        return X_res, y_res

    def _perform_eda(self, df: pd.DataFrame):
        with open("plots/EDA_train.txt", "w") as f:
            print(df.groupby("cnv_type")["means"].describe(), file=f)
            print(df.groupby("cnv_type")["std"].describe(), file=f)
            print(df.groupby("cnv_type")["intq"].describe(), file=f)
            print(df.sort_values(by="means", ascending=False).head(15), file=f)

    def _create_train_test_folders(self):
        os.makedirs(TRAIN_FOLDER, exist_ok=True)
        os.makedirs(TEST_FOLDER, exist_ok=True)
