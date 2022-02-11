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
from dataset import SD_198
from resnet import resnet18
from collections import OrderedDict


epochs = 200
# epochs = 2
lr = 1e-2
milestones = [100, 150]
lrate_decay = 0.1
batch_size = 32
weight_decay = 1e-5
num_workers = 8
optim_type = "sgd"


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

    logfilename = log_dir + 'sgd198-{}-{}-{}-{}'.format(net_type, optim_type, batch_size, lr)

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
        T.RandomResizedCrop(224,scale=(0.3,1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.24705882352941178),
        T.ToTensor(),
        T.Normalize(mean=[0.5816372, 0.46805477, 0.4404385], std=[0.2744409, 0.2529034, 0.25408584]),
        ])

    TEST_TRANSFORMS = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5816372, 0.46805477, 0.4404385], std=[0.2744409, 0.2529034, 0.25408584]),
        ])

    train_dataset = SD_198(root=train_root, train=True, transform=TRAIN_TRANSFORMS)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=False)

    test_dataset = SD_198(root=test_root, train=False, transform=TEST_TRANSFORMS)
    print(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=False)

    model = resnet18(num_classes=train_dataset.class_num, pretrained=False).to(device)

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

    torch.save(pretrained_dict, "./saved_parameters/sd198_resnet18_pretrained_1.pth")

    
if __name__ == "__main__":
    train_root = os.environ["SD198DATASETS"]
    test_root = os.environ["SD198DATASETS"]
    net_type = "resnet18"
    
    run(lr, epochs, batch_size, optim_type, weight_decay, milestones, train_root, test_root, net_type)