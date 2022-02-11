import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SD_198(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.class_num = 158

        self.data, self.targets = self.getdata()

    def getdata(self):
        if self.train:
            txt = 'other_classes_split/train_1.txt'
        else:
            txt = 'other_classes_split/val_1.txt'

        fn = os.path.join(self.root, txt)
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                data.append(os.path.join(self.root, 'images', temp[0]))
                targets.append(int(temp[1]))
        return data, targets

    def __getitem__(self, idx):
        #idx遍历时会使用，并且保证遍历时不会重复，确保在getitem读数据
        path = self.data[idx]
        target = self.targets[idx]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def pil_loader(path):
    """Image Loader
    """
    with open(path, "rb") as afile:
        img = Image.open(afile)
        return img.convert("RGB")


if __name__ == '__main__':
    trainset = SD_198(root=os.environ["SD198DATASETS"], train=True, transform=None)
    print(len(trainset))
    testset = SD_198(root=os.environ["SD198DATASETS"], train=False, transform=None)
    print(len(testset))
    # for item in trainset:
    #     print(item[0], item[1])
