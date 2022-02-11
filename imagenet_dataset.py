import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class imagenet200(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.class_num = 158

        self.data, self.targets = self.getdata()

    def getdata(self):
        if self.train:
            fn = './imagenet200_train.txt'
        else:
            fn = './imagenet200_val.txt'

        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                data.append(os.path.join(self.root, temp[0]))
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
        img = img.convert('RGB')
        img = img.resize((32, 32))
        return img


if __name__ == '__main__':
    trainset = imagenet200(root=os.environ["IMAGENETDATASET"], train=True, transform=None)
    print(len(trainset))
    testset = imagenet200(root=os.environ["IMAGENETDATASET"], train=False, transform=None)
    print(len(testset))
    # for item in trainset:
    #     print(item[0], item[1])