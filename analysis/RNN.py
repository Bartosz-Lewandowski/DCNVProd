import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

TRAIN_FOLDER = "../train"
TEST_FOLDER = "../test"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TRAIN_PATH = "/".join([TRAIN_FOLDER, TRAIN_FILE])
TEST_PATH = "/".join([TEST_FOLDER, TEST_FILE])

lbl_e = LabelEncoder()
columns_to_drop = [
    "chr",
    "start",
    "end",
    "cnv_type",
    "intq",
    "overlap",
    "BAM_CMATCH",
]


def _load_train_files(
    lbl_e: LabelEncoder, columns_to_drop: list
) -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(TRAIN_PATH, sep=",")
    train["cnv_type"] = lbl_e.fit_transform(train["cnv_type"])
    X = train.drop(columns_to_drop, axis=1)
    y = train["cnv_type"]
    return X.head(5000), y.head(5000)


def _load_test_files(
    lbl_e: LabelEncoder, columns_to_drop: list
) -> tuple[pd.DataFrame, pd.Series]:
    df_test = pd.read_csv(TEST_PATH, sep=",")
    df_test["cnv_type"] = lbl_e.transform(df_test["cnv_type"])
    X_test_sim = df_test.drop(columns_to_drop, axis=1)
    y_test_sim = df_test["cnv_type"]
    return X_test_sim, y_test_sim


X, y = _load_train_files(lbl_e, columns_to_drop)


# Assuming X is your feature matrix and y is your labels
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define your LSTM model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.lstm = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.RNN(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)  # Take the output from the last time step
        return out


# Define the model
input_size = X_train.shape[1]  # Number of features
hidden_size = 64
num_classes = len(np.unique(y_train))
model = SimpleLSTM(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
