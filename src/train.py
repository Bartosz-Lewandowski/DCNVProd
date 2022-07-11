import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib 
from collections import OrderedDict
import matplotlib.pyplot as plt

models = [
    (
        "RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(
            oob_score=True,
            max_features="sqrt",
            random_state=123,
        ),
    ),
    (
        "RandomForestClassifier, max_features=None",
        RandomForestClassifier(
            max_features=None,
            oob_score=True,
            random_state=123,
        ),
    ),
]
training_in = pd.read_csv("results.bed",header=None,sep="\t")
X = training_in.drop(training_in[[0,1,2,3,4]], axis=1)
X.columns = list(range(0,len(X.columns)))
Y = list(training_in[3])
# clf = models["RFC100"]
min_estimators = 100
max_estimators = 300
error_rate = OrderedDict((label, []) for label, _ in models)

for label, clf in models:
    for i in range(min_estimators, max_estimators + 1, 100):
        print(i)
        clf.set_params(n_estimators=i)
        clf.fit(X, Y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)
    plt.savefig("results.png")


# clf.fit(X,Y)
# Y2 = list(map(str,list(training_in[4])))
# clf2 = RandomForestClassifier(n_estimators=100, oob_score = True)
# clf2.fit(X,Y2)
# joblib.dump(clf, "model")
# joblib.dump(clf2, "model" + "2")