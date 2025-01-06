"""## Define Transformations for Images and Targets"""

import torchvision.transforms.functional as TF
import random

class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, target):
        if random.random() < self.probability:
            image = TF.hflip(image)
            width = image.shape[2]
            boxes = target['boxes']
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target['boxes'] = boxes
        return image, target

def get_transform(train):
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return ComposeTransforms(transforms)