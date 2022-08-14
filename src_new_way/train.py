import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


def train_data(data):
    df = pd.read_csv(data, sep=",", dtype={"chr": object})
    he = pd.get_dummies(df["cnv_type"])
    df_new = pd.concat([df, he], axis=1)
    lbl_e = LabelEncoder()
    df["cnv_type"] = lbl_e.fit_transform(df["cnv_type"])
    X = df.drop(["chr", "start", "end", "cnv_type"], axis=1)
    y = df["cnv_type"]
    feature_names = X.columns
    forest = RandomForestClassifier(random_state=0, n_estimators=200, oob_score=True)
    forest.fit(X, y)
    pred_train = np.argmax(forest.oob_decision_function_, axis=1)
    print(len(pred_train))
    print(f1_score(y, pred_train, average="weighted"))
    print(precision_score(y, pred_train, average="weighted"))
    print(recall_score(y, pred_train, average="weighted"))
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.savefig("feature_importances.png")

    # Using Pearson Correlation
    plt.figure(figsize=(12, 10))
    cor = df_new.drop(["chr", "start", "end", "cnv_type"], axis=1).corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig("correlation.png")
