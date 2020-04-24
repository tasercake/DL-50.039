#!/usr/bin/env python
import numpy as np
import sklearn
from pathlib import Path
from sklearn.svm import SVC


CONCEPTS_FILE = Path("dataset/classes.txt")
DATASET_DIR = Path("dataset/")


def one_vs_rest(c, x, y, x_test, y_test, *args, **kwargs):
    acc = []
    for i in range(y.shape[1]):
        clf = SVC(C=C, *args, **kwargs)
        clf.fit(x, y[:, i])
        acc.append(clf.score(x_test, y_test[:, i]))
    return np.mean(acc)


if __name__ == "__main__":
    # Get class names
    with CONCEPTS_FILE.open("r") as f:
        concepts = [line.rstrip() for line in f.readlines()]
        print(f"Discovered {len(concepts)} concepts: {concepts}")

    # load data
    x_train = np.load(DATASET_DIR / "x_train.npy")
    y_train = np.load(DATASET_DIR / "y_train.npy")
    x_val = np.load(DATASET_DIR / "x_val.npy")
    y_val = np.load(DATASET_DIR / "y_val.npy")
    x_test = np.load(DATASET_DIR / "x_test.npy")
    y_test = np.load(DATASET_DIR / "y_test.npy")

    # Determine best C
    acc = {}
    for C in [0.01, 0.1, 0.1 ** 0.5, 1, 10 ** 0.5, 10, 100]:
        # acc[C] = one_vs_rest(SVC(kernel="linear", C=C), x_train, y_train, x_val, y_val)
        acc[C] = one_vs_rest(C, x_train, y_train, x_val, y_val, kernel="rbf")
        print(f"Mean acc @ C={C:.3f}: {acc[C]:.4f}")
    best_C = max(acc, key=acc.get)
    print(f"best C = {best_C:.3f}")

    # Fit on train + val, evaluate on test set
    final_acc = one_vs_rest(
        best_C,
        np.vstack((x_train, x_val)),
        np.vstack((y_train, y_val)),
        np.vstack((x_test, x_test)),
        np.vstack((y_test, y_test)),
        kernel="rbf",
    )
    print(f"Test set accuracy @ C={best_C:.3f}: {final_acc:.4f}")
