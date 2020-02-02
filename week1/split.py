#!/usr/bin/env python
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from pathlib import Path
from joblib import Parallel, delayed


if __name__ == "__main__":
    imageclef_dir = Path("imageclef2011_feats")
    labels_file = Path("trainset_gt_annotations.txt")

    with labels_file.open("r") as f:
        lines = [line.rstrip().split() for line in f.readlines()]

    labels_map = {l[0]: [int(i) for i in l[1:]] for l in lines}
    features_map = dict(
        Parallel(n_jobs=-1)(
            delayed(
                lambda name: (name, np.load(list(imageclef_dir.glob(name + "*"))[0]))
            )(name)
            for name in labels_map
        )
    )
    features = np.array([features_map[name] for name in sorted(labels_map.keys())])
    labels = np.array([labels_map[name] for name in sorted(labels_map.keys())])
    x_train, y_train, x_test, y_test= iterative_train_test_split(features, labels, 0.25)
    
    print(labels.sum(axis=0) / labels.sum())
    print(y_train.sum(axis=0) / y_train.sum())
    print(y_test.sum(axis=0) / y_test.sum())
