import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

# import optuna
from .metrics import CNVMetric
from .paths import TEST_PATH, TRAIN_PATH


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.hidden(x)
        out = self.relu(out)
        out = self.output(out)
        return out


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
        self.stats1 = True
        self.stats2 = True
        self.bam_fc = True
        self.prev_and_next = True
        self.log = False
        self.best_score: float = 0.0
        self.mapping = {"del": 0, "dup": 1, "normal": 2}

    def train(self):
        X, y = self._load_train_files()
        train_data = self.convert_data_to_pytorch_dataset(X, y)
        train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        input_size = X.shape[1]  # Number of features
        hidden_size = 64
        num_classes = len(np.unique(y))
        self.model = DNN(input_size, hidden_size, num_classes)
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        num_epochs = 2
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for batch_x, batch_y in tqdm.tqdm(
                train_data_loader, total=len(train_data_loader)
            ):
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def evaluate_on_test_data(self):
        X_test, y_test = self._load_test_files()
        test_data = self.convert_data_to_pytorch_dataset(X_test, y_test)
        dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_x, batch_y in tqdm.tqdm(dataloader, total=len(dataloader)):
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(batch_y.tolist())
                y_pred.extend(predicted.tolist())

            test_data = pd.read_csv(TEST_PATH, sep=",")
            test_data["pred"] = self.map_function(y_pred, reverse_mapping=True)
            cnv = CNVMetric(test_data)
            with open("results/simple_DNN_model.txt", "w") as f:
                f.write("Simple DNN Model\n")
                f.write(f"F1 Score: {f1_score(y_true, y_pred, average='macro')}")
                f.write(str(confusion_matrix(y_true, y_pred)))
                f.write(classification_report(y_true, y_pred))
                f.write(f"CNV Metric: {cnv.get_metrics()}")

    def _load_train_files(self) -> tuple[pd.DataFrame, pd.Series]:
        train = pd.read_csv(TRAIN_PATH, sep=",")
        if self.eda:
            self._perform_eda(train)
        train["cnv_type"] = self.map_function(train["cnv_type"])
        X = train.drop(self.columns_to_drop, axis=1)
        y = train["cnv_type"]
        return X, y

    def _load_test_files(self) -> tuple[pd.DataFrame, pd.Series]:
        df_test = pd.read_csv(TEST_PATH, sep=",")
        df_test["cnv_type"] = self.map_function(df_test["cnv_type"])
        X_test_sim = df_test.drop(self.columns_to_drop, axis=1)
        y_test_sim = df_test["cnv_type"]
        return X_test_sim, y_test_sim

    def convert_data_to_pytorch_dataset(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[TensorDataset, TensorDataset]:
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        tensor_dataset = TensorDataset(X_tensor, y_tensor)
        return tensor_dataset

    def map_function(self, target_values: list, reverse_mapping: bool = False) -> list:
        if reverse_mapping:
            rev_map = {v: k for k, v in self.mapping.items()}
            return list(map(rev_map.get, target_values))
        else:
            return list(map(self.mapping.get, target_values))


# # .... (rest of your code) ...

# def objective(trial):
#     # Hyperparameters suggested by Optuna
#     input_size = X_train.shape[1]
#     hidden_size = trial.suggest_int('hidden_size', 32, 256)
#     num_classes = len(np.unique(y_train))
#     lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
#     num_epochs = trial.suggest_int('num_epochs', 5, 50)
#     batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
#     optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'RMSprop'])
#     activation = trial.suggest_categorical('activation', ['ReLU', 'Sigmoid', 'Tanh'])
#     num_layers = trial.suggest_int('num_layers', 1, 3)
#     weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
#     dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

#     # Model with suggested hyperparameters
#     model = DNN(input_size, hidden_size, num_classes, num_layers, activation, dropout_rate)  # Updated DNN

#     # Instantiate Optimizer
#     if optimizer_name == 'SGD':
#         optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
#     elif optimizer_name == 'Adam':
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     else:
#         optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

#     # ... (Rest of your training and evaluation code) ...

#     # Return evaluation metric
#     evaluate_model(test_loader, model)   # Assuming it already returns the F1-score
#     return f1

# # Hyperparameter Search
# study = optuna.create_study(direction='maximize')  # Maximize F1-score
# study.optimize(objective, n_trials=50)  # Set the number of trials

# print('Best Hyperparameters:', study.best_params)

# # Use the best hyperparameters for the final model ...


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
