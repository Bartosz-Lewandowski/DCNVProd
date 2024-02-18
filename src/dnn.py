import os
from typing import Optional, Union

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, TensorDataset

from .metrics import CNVMetric
from .paths import TEST_PATH, TRAIN_PATH


class DNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_classes, num_layers, activation, dropout_rate
    ):
        super(DNN, self).__init__()

        self.layers = []
        for i in range(num_layers):
            if i == 0:  # Input layer
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:  # Hidden layers
                self.layers.append(nn.Linear(hidden_size, hidden_size))

            # Activation
            if activation == "ReLU":
                self.layers.append(nn.ReLU())
            elif activation == "ELU":
                self.layers.append(nn.ELU())
            elif activation == "SELU":
                self.layers.append(nn.SELU())
            # Dropout
            if dropout_rate > 0.0:
                self.layers.append(nn.Dropout(dropout_rate))

        self.layers.append(nn.Linear(hidden_size, num_classes))  # Output layer

        # Convert layers to a sequential model
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class Train:
    def __init__(self, eda: bool):
        self.eda = eda
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
        self.num_classes = 3
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using {self.device} device")

    def train(self):
        X_train, X_val, y_train, y_val = self._load_train_and_val_files()
        study = optuna.create_study(direction="maximize")  # Maximize F1-score
        study.optimize(
            lambda trial: self._objective(trial, X_train, X_val, y_train, y_val),
            timeout=3600 * 30,
        )
        results_df = pd.DataFrame(self.results)
        self.__save_HP_results()
        self.best_params = (
            results_df.sort_values("f1", ascending=False).iloc[0].to_dict()
        )

        X, y = self._combine_train_and_val(X_train, X_val, y_train, y_val)

        X_preprocessed = self._preprocess_data(X)

        train_data = self.convert_data_to_pytorch_dataset(X_preprocessed, y)
        train_data_loader = DataLoader(
            train_data, batch_size=self.best_params["batch_size"], num_workers=10
        )
        # Model parameters
        input_size = X_preprocessed.shape[1]  # Number of features
        hidden_size = self.best_params["hidden_size"]
        num_epochs = self.best_params["num_epochs"]
        num_layers = self.best_params["num_layers"]
        activation = self.best_params["activation"]
        dropout_rate = self.best_params["dropout_rate"]
        optimizer_name = self.best_params["optimizer"]
        lr = self.best_params["lr"]
        weight_decay = self.best_params["weight_decay"]

        # Model with best hyperparameters
        model = DNN(
            input_size,
            hidden_size,
            self.num_classes,
            num_layers,
            activation,
            dropout_rate,
        ).to(self.device)

        # Optimizer
        optimizer = self.__get_optimizer(optimizer_name, model, lr, weight_decay)

        # Loss function
        weights = self.__get_class_weights(y, self.best_params["class_weight"])
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Train the model
        model = self.__train_loop(
            train_data_loader, model, num_epochs, optimizer, criterion
        )

        # save model to disk
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/DNN_model.pth")

    def _evaluate_model(self):
        X_test, y_test = self._load_test_files()
        if not self.best_params:
            self.best_params = (
                pd.read_csv("HP_results/DNN_HPS_results.csv")
                .sort_values("f1", ascending=False)
                .iloc[0]
                .to_dict()
            )

        X_test_preprocessed = self._preprocess_data(X_test)
        test_data = self.convert_data_to_pytorch_dataset(X_test_preprocessed, y_test)
        test_data = DataLoader(test_data, batch_size=self.best_params["batch_size"])

        input_size = X_test_preprocessed.shape[1]  # Number of features
        hidden_size = self.best_params["hidden_size"]
        num_layers = self.best_params["num_layers"]
        activation = self.best_params["activation"]
        dropout_rate = self.best_params["dropout_rate"]

        model = DNN(
            input_size,
            hidden_size,
            self.num_classes,
            num_layers,
            activation,
            dropout_rate,
        ).to(self.device)
        model.load_state_dict(torch.load("models/DNN_model.pth"))

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_x, batch_y in tqdm.tqdm(test_data, total=len(test_data)):
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(batch_y.tolist())
                y_pred.extend(predicted.tolist())

            test_data = pd.read_csv(TEST_PATH, sep=",")
            test_data["pred"] = self.map_function(y_pred, reverse_mapping=True)
            cnv = CNVMetric(test_data)
            with open("results/simple_DNN_model.txt", "w") as f:
                f.write("Simple DNN Model\n")
                f.write(f"F1 Score: {f1_score(y_true, y_pred, average='macro')}\n")
                f.write(f"{str(confusion_matrix(y_true, y_pred))}\n")
                f.write(f"{classification_report(y_true, y_pred)}\n")
                f.write(f"CNV Metric: {cnv.get_metrics()}")

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> float:
        class_weights = trial.suggest_categorical(
            "class_weights",
            [None, "balanced", [3, 3, 1], [4, 4, 1], [3, 5, 1]],
        )
        scaler = trial.suggest_categorical("scaler", ["log", None])
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
        else:
            X_val_res = X_val

        hidden_size = trial.suggest_int("hidden_size", 32, 512, step=32)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        num_epochs = trial.suggest_int("num_epochs", 2, 20)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 512])
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["SGD", "Adam", "Nadam", "RMSprop"]
        )
        activation = trial.suggest_categorical("activation", ["ReLU", "ELU", "SELU"])
        num_layers = trial.suggest_int("num_layers", 1, 8)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

        # Convert data to PyTorch Dataset
        train_dataset = self.convert_data_to_pytorch_dataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10)
        val_dataset = self.convert_data_to_pytorch_dataset(X_val_res, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10)

        input_size = X_train.shape[1]  # Number of features
        # Model with suggested hyperparameters
        model = DNN(
            input_size,
            hidden_size,
            self.num_classes,
            num_layers,
            activation,
            dropout_rate,
        )

        # Optimizer
        optimizer = self.__get_optimizer(optimizer_name, model, lr, weight_decay)

        # Loss function
        weights = self.__get_class_weights(y_train, class_weights)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Train the model
        model = self.__train_loop(train_loader, model, num_epochs, optimizer, criterion)

        # Evaluate the model
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_x, batch_y in tqdm.tqdm(val_loader, total=len(val_loader)):
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(batch_y.tolist())
                y_pred.extend(predicted.tolist())
            f1 = f1_score(y_true, y_pred, average="macro")

        self.results.append(
            {
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "lr": lr,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "activation": activation,
                "weight_decay": weight_decay,
                "dropout_rate": dropout_rate,
                "class_weight": class_weights,
                "scaler": scaler,
                "stats1": stats1,
                "stats2": stats2,
                "bam_fc": bam_fc,
                "prev_and_next": prev_and_next,
                "f1": f1,
            }
        )

        return f1

    def __train_loop(
        self,
        train_loader: DataLoader,
        model: DNN,
        num_epochs: int,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ):
        for epoch in range(num_epochs):
            for batch_x, batch_y in tqdm.tqdm(train_loader, total=len(train_loader)):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        return model

    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.best_params["stats1"]:
            X = X.drop(["STAT_CROSS"], axis=1)

        if not self.best_params["stats2"]:
            X = X.drop(["STAT_CROSS2"], axis=1)

        if not self.best_params["bam_fc"]:
            X = X.drop(["BAM_CROSS"], axis=1)

        if not self.best_params["prev_and_next"]:
            X = X.drop(["PR_5", "NXT_5", "PR_10", "NXT_10", "PR_20", "NXT_20"], axis=1)

        if self.best_params["scaler"] == "log":
            X = np.log1p(X)

        return X

    def _load_train_and_val_files(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        train = pd.read_csv(TRAIN_PATH, sep=",")
        val = train[train["chr"].isin([9])]
        train = train[~train["chr"].isin([9])]
        train["cnv_type"] = self.map_function(train["cnv_type"])
        val["cnv_type"] = self.map_function(val["cnv_type"])
        X_train, X_val = train.drop(self.columns_to_drop, axis=1), val.drop(
            self.columns_to_drop, axis=1
        )
        y_train, y_val = train["cnv_type"], val["cnv_type"]
        return X_train, X_val, y_train, y_val

    def _load_test_files(self) -> tuple[pd.DataFrame, pd.Series]:
        df_test = pd.read_csv(TEST_PATH, sep=",")
        df_test["cnv_type"] = self.map_function(df_test["cnv_type"])
        X_test_sim = df_test.drop(self.columns_to_drop, axis=1)
        y_test_sim = df_test["cnv_type"]
        return X_test_sim, y_test_sim

    def _combine_train_and_val(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.concat([X_train, X_val], axis=0)
        y = pd.concat([y_train, y_val], axis=0)
        return X, y

    def convert_data_to_pytorch_dataset(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[TensorDataset, TensorDataset]:
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        tensor_dataset = TensorDataset(X_tensor, y_tensor)
        return tensor_dataset

    def __get_optimizer(
        self, optimizer_name: str, model: DNN, lr: float, weight_decay: float
    ) -> optim.Optimizer:
        if optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "Nadam":
            optimizer = optim.NAdam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            optimizer = optim.RMSprop(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        return optimizer

    def __get_class_weights(
        self, y: pd.Series, weights: Optional[Union[list, str]]
    ) -> torch.Tensor:
        if weights == "balanced":
            class_weights = torch.tensor(
                class_weight.compute_class_weight(
                    class_weight="balanced", classes=np.unique(y), y=y
                ),
                dtype=torch.float32,
            ).to(self.device)
        elif isinstance(weights, list):
            class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            class_weights = None
        return class_weights

    def map_function(self, target_values: list, reverse_mapping: bool = False) -> list:
        if reverse_mapping:
            rev_map = {v: k for k, v in self.mapping.items()}
            return list(map(rev_map.get, target_values))
        else:
            return list(map(self.mapping.get, target_values))

    def __log_transform(
        self, X_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train = np.log1p(X_train)
        x_test = np.log1p(x_test)
        return X_train, x_test

    def __save_HP_results(self):
        results_df = pd.DataFrame(self.results)
        os.makedirs("HP_results", exist_ok=True)
        results_df.to_csv("HP_results/DNN_HPS_results.csv", index=False)


# if __name__ == "__main__":
#     lbl_e = LabelEncoder()
#     X, y = _load_train_files(lbl_e, columns_to_drop)
#     X_test, y_test = _load_test_files(lbl_e, columns_to_drop)
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     train_dataset = convert_data_to_pytorch_dataset(X_train, y_train)
#     val_dataset = convert_data_to_pytorch_dataset(X_val, y_val)
#     test_dataset = convert_data_to_pytorch_dataset(X_test, y_test)
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#     # Define the model
#     input_size = X_train.shape[1]  # Number of features
#     hidden_size = 64
#     num_classes = len(np.unique(y_train))
#     model = DNN(input_size, hidden_size, num_classes)

#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001)
#     num_epochs = 10

#     # Train the model
#     train_model(train_loader, model, num_epochs, optimizer)

#     # Evaluate the model
#     evaluate_model(test_loader, model)
