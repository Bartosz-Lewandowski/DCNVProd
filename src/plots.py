import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_count(feature, data, name):
    plt.figure(figsize=(7, 7))
    ax = sns.countplot(x=feature, data=data, palette="colorblind")
    height = sum([p.get_height() for p in ax.patches])
    for p in ax.patches:
        ax.annotate(
            f"{100*p.get_height()/height:.2f} %",
            (p.get_x() + 0.3, p.get_height() + 5),
            animated=True,
        )
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_yticklabels([int(x / 1000000) for x in ticks_loc])
    # ax.set_xticklabels(["Brak CNV", "Delecja", "Duplikacja"])
    ax.set_xlabel("Typ CNV")
    ax.set_ylabel("Liczba okien genomowych wyrażona w milionach")
    os.makedirs("plots/", exist_ok=True)
    plot_name = f"plots/{feature}_{name}_plot.png"
    plt.savefig(plot_name)


def feature_correlation(X):
    # Using Spearman Correlation
    plt.figure(figsize=(12, 12))
    cor = X.corr(method="spearman").round(2)
    mask = np.triu(np.ones_like(cor, dtype=bool))
    mask = mask[1:, :-1]
    corr = cor.iloc[1:, :-1].copy()
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
        annot=True,
        square=True,
    )
    yticks = [i.upper() for i in corr.index]
    xticks = [i.upper() for i in corr.columns]
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    plt.xticks(plt.xticks()[0], labels=xticks)
    # title
    title = "FEATURE CORRELATION MATRIX"
    plt.title(title, fontsize=18)

    plt.savefig("correlation.png")


def feature_importances(feature_names, importances):
    forest_importances = pd.Series(importances, index=feature_names)
    plt.figure(figsize=(14, 10))
    palette = sns.diverging_palette(230, 0, 90, 60, n=19)
    sns.barplot(
        x=forest_importances.index, y=forest_importances.values, palette=palette
    )
    xticks = [i.upper() for i in forest_importances.index]
    plt.xticks(plt.xticks()[0], labels=xticks, rotation=40)
    plt.savefig("feature_importances.png")


def confusion_matrix_plot(true, y_pred):
    cm = confusion_matrix(true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", ax=ax)
    ax.set_xlabel("Przewidziane klasy")
    ax.set_ylabel("Faktyczne klasy")
    ax.set_title("Macierz błędu")
    ax.xaxis.set_ticklabels(["0", "1", "2"])
    ax.yaxis.set_ticklabels(["0", "1", "2"])
    plt.plot()
