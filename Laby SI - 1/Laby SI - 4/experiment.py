"""Test accuracy on Titanic over many random 80/20 splits, compared with two baselines.

One split of ~140 test passengers is too small to rank models (one passenger = 0.7 points),
so every model is trained and tested on the same SPLITS splits and the mean and spread are reported.

usage: python experiment.py [number of splits]
"""
import sys
import numpy as np

from decision_tree import DecisionTree
from random_forest import RandomForest
from load_data import load_titanic

SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEX = 5  #column of the 'Sex' feature (1 = female) in load_titanic's X


def accuracy(predicted, y):
    return np.mean(np.round(predicted) == y)


def run(seed):
    np.random.seed(seed)
    (X_train, y_train), (X_test, y_test) = load_titanic()
    majority = round(np.mean(y_train))
    results = {
        "Majority class": accuracy(np.full(len(y_test), majority), y_test),
        "Women survive, men don't": accuracy(X_test[:, SEX], y_test),
    }
    tree = DecisionTree({"depth": 14})
    tree.train(X_train, y_train)
    results["Decision tree (depth 14)"] = accuracy(tree.predict(X_test), y_test)
    forest = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 14})
    forest.train(X_train, y_train)
    results["Random forest (10 trees, 2 features)"] = accuracy(forest.predict(X_test), y_test)
    return results


if __name__ == "__main__":
    runs = [run(seed) for seed in range(SPLITS)]
    print(f"Test accuracy over {SPLITS} random splits (mean ± standard deviation, min - max):")
    for name in runs[0]:
        values = np.array([r[name] for r in runs])
        print(f"  {name:<38} {values.mean():.3f} ± {values.std():.3f}   ({values.min():.3f} - {values.max():.3f})")
