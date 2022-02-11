import os
import numpy as np


if __name__ == "__main__":
    classes = []
    train_dir = os.path.join(os.environ["IMAGENETDATASET"], "train")
    val_dir = os.path.join(os.environ["IMAGENETDATASET"], "val")

    # f = open("./imagenet_pretrain.txt", "r")
    with open("imagenet_pretrain.txt") as f:
        for line in f.readlines():
            class_name = line.strip().split("/")[-2]
            classes.append(class_name)
    classes = np.unique(classes)
    classes.sort()

    train_pics = []
    for idx, item in enumerate(classes):
        for img_name in os.listdir(os.path.join(train_dir, item)):
            if ".JPEG" in img_name:
                train_pics.append(os.path.join(item, img_name) + " " + str(idx))
    
    print(len(train_pics))
    
    with open("imagenet200_train.txt", "w") as f:
        f.write("\n".join(train_pics))
    
    val_pics = []
    for idx, item in enumerate(classes):
        for img_name in os.listdir(os.path.join(val_dir, item)):
            if ".JPEG" in img_name:
                val_pics.append(os.path.join(item, img_name) + " " + str(idx))
    
    print(len(val_pics))
    
    with open("imagenet200_val.txt", "w") as f:
        f.write("\n".join(val_pics))

    
    # print(len(classes))