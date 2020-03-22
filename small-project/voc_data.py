#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
import skimage
from skimage import io
from operator import itemgetter
from more_itertools import unique_everseen
from natsort import natsorted
from PIL import Image

import torch
from torch.utils.data import Dataset
# from bs4 import BeautifulSoup
# import matplotlib.pyplot as plt


class PascalVOC:
    """
    Handle Pascal VOC dataset
    """

    def __init__(self, root_dir):
        """
        Summary:
            Init the class with root dir
        Args:
            root_dir (str | Path): path to your voc dataset
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "JPEGImages"
        self.ann_dir = self.root_dir / "Annotations"
        self.set_dir = self.root_dir / "ImageSets" / "Main"
        self.cache_dir = self.root_dir / "csvs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_image_sets(self):
        """
        Summary:
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    def _imgs_from_category(self, cat_name, dataset):
        """
        Summary:
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = self.set_dir.joinpath(f"{cat_name}_{dataset}.txt")
        df = pd.read_csv(
            filename, delim_whitespace=True, header=None, names=["filename", "true"]
        )
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary:
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        df = df[df["true"] == 1]
        return df["filename"].values.tolist()


def invert_dict(d):
    inverted = {}
    for k, v in d.items():
        for x in v:
            inverted.setdefault(x, []).append(k)
    return inverted


class VOCDataset(Dataset):
    def __init__(self, voc, split="train", transform=None):
        self.voc = voc
        self.split = split
        self.transform = transform
        self.categories = self.voc.list_image_sets()
        self.num_cats = len(self.categories)
        self.cat_to_index = {cat: i for i, cat in enumerate(self.categories)}

        _samples = {}
        for cat in self.categories:
            cat_files = self.voc.imgs_from_category_as_list(cat, self.split)
            cat_files = [
                self.voc.img_dir.joinpath(f).with_suffix(".jpg") for f in cat_files
            ]
            _samples[cat] = cat_files
        _inverted = invert_dict(_samples)
        # TODO: include images that have no objects in them
        # for file in self.voc.img_dir.iterdir():
        #     _inverted.setdefault(file, [])
        try:
            assert all((f.exists for f in _inverted))
        except AssertionError:
            raise ValueError("Failed to find some or all image files")
        self.image_to_categories = {
            file: [self.cat_to_index[cat] for cat in cats]
            for file, cats in _inverted.items()
        }
        # self.image_to_categories = {
        #     file: np.eye(self.num_cats, dtype=bool)[cats].sum(axis=0).astype(np.float32)
        #     if cats
        #     else np.zeros(self.num_cats, dtype=np.float32)
        #     for file, cats in self.image_to_categories.items()
        # }
        self.samples = list(self.image_to_categories.items())

    @staticmethod
    def _one_hot(labels):
        hot = torch.zeros(20, dtype=torch.float32)
        for l in labels:
            hot[l] = 1
        return hot

    def __getitem__(self, idx):
        file, labels = self.samples[idx]
        image = Image.open(file)
        if self.transform:
            image = self.transform(image)
        return image, self._one_hot(labels)

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    pv = PascalVOC("./data/VOCdevkit/VOC2012/")
    # cat_name = "car"
    # dataset = "val"
    # ls = pv.imgs_from_category_as_list(cat_name, dataset)
    # print(len(ls), ls[0])
    dset = VOCDataset(pv)
    print(len(dset))
    print(dset[800][1])
