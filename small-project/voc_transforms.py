import torch
from torchvision import transforms


class ImageOnly:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform(sample[0]), sample[1]


class LabelsOnly:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return sample[0], self.transform(sample[1])
