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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


# Define the objective function to optimize
def objective(trial):
    # Define the hyperparameters to search over
    model_type = trial.suggest_categorical(
        "model_type", ["XGBoost", "LightGBM", "RandomForest"]
    )
    max_depth = trial.suggest_int("max_depth", 20, 100, step=20)
    class_weight = trial.suggest_categorical(
        "class_weight", [None, "balanced", {0: 3, 1: 3, 2: 1}]
    )
    scaler = trial.suggest_categorical("scaler", ["StandardScaler", "log", None])
    undersampling = trial.suggest_categorical("undersampling", [True, False])

    # Preprocess the data based on hyperparameters
    if undersampling:
        # Undersampling klas mniejszościowych (przykład)
        under = RandomUnderSampler(sampling_strategy="majority", random_state=42)
        X_res, y_res = under.fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train

    if scaler == "StandardScaler":
        s = StandardScaler()
        X_res = s.fit_transform(X_res)
        x_test_res = s.transform(X_test)
    elif scaler == "log":
        X_res = np.log1p(X_res)
        x_test_res = np.log1p(X_test)
    else:
        x_test_res = X_test

    if model_type == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 20, 260, step=20)
        params = {
            "min_samples_split": trial.suggest_int("min_samples_split", 1, 150),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
        }
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params,
        )
    elif model_type == "LightGBM":
        n_estimators = trial.suggest_int("n_estimators", 20, 260, step=20)
        params = {
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]
            ),
            "num_leaves": trial.suggest_int("num_leaves", 50, 700),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 5000),
            "verbosity": -1,
        }
        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params,
        )
    elif model_type == "XGBoost":
        n_estimators = trial.suggest_int("n_estimators", 20, 80, step=20)
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
            params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 0.4, log=True)
            params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 0.4, log=True)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_pos_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params,
        )

    # Trenowanie modelu
    model.fit(X_res, y_res)

    # Przewidywanie na zbiorze testowym

    y_pred = model.predict(x_test_res)
    fbeta = fbeta_score(y_test, y_pred, beta=3, average="macro")

    results.append(
        {
            "model": model_type,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": class_weight,
            "scaler": scaler,
            "undersampling": undersampling,
            "params": params,
            "fbeta": fbeta,
            "classification_report": classification_report(
                y_test, y_pred, zero_division=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
    )

    return fbeta


if __name__ == "__main__":
    df = pd.read_csv("../stats/combined.csv", sep=",", dtype={"chr": object})
    test = df[df["chr"] == "13"]
    train = df[df["chr"] != "13"]
    lbl_e = LabelEncoder()
    y_train = lbl_e.fit_transform(train["cnv_type"])
    X_train = train.drop(
        [
            "chr",
            "start",
            "end",
            "cnv_type",
        ],
        axis=1,
    )
    y_test = lbl_e.transform(test["cnv_type"])
    X_test = test.drop(
        [
            "chr",
            "start",
            "end",
            "cnv_type",
        ],
        axis=1,
    )
    results: list = []
    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=3600 * 24)

    # Get the best hyperparameters
    best_params = study.best_params
    best_accuracy = study.best_value

    print(f"Best hyperparameters: \n{best_params}")
    print(f"Best fbeta value: \n{best_accuracy}")

    pd.DataFrame(results).to_csv("results.csv", index=False)
