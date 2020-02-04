#!/usr/bin/env python
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

CONCEPTS_FILE = Path("concepts_2011.txt")
IMAGECLEF_DIR = Path("imageclef2011_feats")
LABELS_FILE = Path("trainset_gt_annotations.txt")
OUTPUT_DIR = Path("dataset")
TRAIN = 0.6
VAL = 0.15
TEST = 0.25
CLASSES = ["Spring", "Summer", "Autumn", "Winter"]


def load_concepts(file):
    with open(file, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]
        concepts = {
            int(index): name for index, name in [line.split() for line in lines[1:]]
        }
        concept_indices = {v: k for k, v in concepts.items()}
        return concepts, concept_indices


def load_imageclef_dataset(imageclef_dir, labels_file):
    with labels_file.open("r") as f:
        lines = [line.rstrip().split() for line in f.readlines()][:800]

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
    return features, labels


if __name__ == "__main__":
    concepts, concept_indices = load_concepts(CONCEPTS_FILE)
    features, labels = load_imageclef_dataset(IMAGECLEF_DIR, LABELS_FILE)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "classes.txt", "w") as f:
        f.write("\n".join(CLASSES) + "\n")

    dataset = defaultdict(lambda: [])
    for name in CLASSES:
        index = concept_indices[name]
        x, y = features[:, index], labels[:, index]

        train_ratio = TRAIN
        val_ratio = VAL / (1 - train_ratio)
        test_ratio = TEST / (1 - train_ratio)

        x_train, x_rest, y_train, y_rest = train_test_split(
            x, y, train_size=train_ratio, stratify=y
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_rest, y_rest, train_size=val_ratio, stratify=y_rest
        )
        dataset["x_train"].append(x_train)
        dataset["x_val"].append(x_val)
        dataset["x_test"].append(x_test)
        dataset["y_train"].append(y_train)
        dataset["y_val"].append(y_val)
        dataset["y_test"].append(y_test)

    dataset = {k: np.array(v) for k, v in dataset.items()}
    for filename, subset in dataset.items():
        np.save(OUTPUT_DIR / filename, subset)
    print({k: v.shape for k, v in dataset.items()})
