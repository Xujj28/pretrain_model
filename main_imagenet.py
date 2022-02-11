import imp
import torch
import torch.nn as nn
import os
import torchvision.transforms as T
import logging
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from imagenet_dataset import imagenet200
from cifar_resnet_cbam import resnet18_cbam
from collections import OrderedDict


epochs = 101
# epochs = 2
lr = 1e-3
milestones = [45, 90]
lrate_decay = 0.1
batch_size = 64
weight_decay = 2e-4
num_workers = 8
optim_type = "adam"


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def compute_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)['logits']
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == targets).sum()
        total += len(targets)

    return np.around(tensor2numpy(correct)*100 / total, decimals=2)


def run(lr, epochs, batch_size, optim_type, weight_decay, milestones, train_root, test_root, net_type, dict_path=None):
    device = 'cuda:0'

    log_dir = "./log/"

    logfilename = log_dir + 'imagenet200-{}-{}-{}-{}'.format(net_type, optim_type, batch_size, lr)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("lr:{}\n epochs:{}\n batch_size:{}\n optim_type:{}\n weight_decay:{}\n milestones:{}\n train_root:{}\n test_root:{}\n net_type:{}\n"
    .format(lr, epochs, batch_size, optim_type, weight_decay, milestones, train_root, test_root, net_type))

    TRAIN_TRANSFORMS = T.Compose([
        T.RandomCrop((32,32),padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.24705882352941178),
        T.ToTensor(),
        T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ])

    TEST_TRANSFORMS = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ])

    train_dataset = imagenet200(root=train_root, train=True, transform=TRAIN_TRANSFORMS)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=False)

    test_dataset = imagenet200(root=test_root, train=False, transform=TEST_TRANSFORMS)
    print(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=False)

    model = resnet18_cbam(num_classes=train_dataset.class_num, pretrained=False).to(device)

    if optim_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
    criterion = nn.CrossEntropyLoss()

    prog_bar = tqdm(range(epochs))
    for _, epoch in enumerate(prog_bar):
        model.train()
        losses = 0.
        correct, total = 0, 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)['logits']
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            # acc
            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)

        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        test_acc = compute_accuracy(model, test_loader, device)
        info = 'Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
            epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
        prog_bar.set_description(info)

        logging.info(info)
    
    pretrained_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if "fc" not in k:
            pretrained_dict[k] = v

    torch.save(pretrained_dict, "./saved_parameters/imagenet200_resnet18_cbam_pretrained.pth")

    
if __name__ == "__main__":
    train_root = os.path.join(os.environ["IMAGENETDATASET"], "train")
    test_root = os.path.join(os.environ["IMAGENETDATASET"], "val")
    net_type = "resnet18_cbam"
    
    run(lr, epochs, batch_size, optim_type, weight_decay, milestones, train_root, test_root, net_type)