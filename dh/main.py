from torchvision import models
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.utils.prune as prune
from tqdm import tqdm
import copy
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import random


# reproduce
def reproduce(seed):
    seed = seed
    deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, trainloader, optimizer, criterion, scheduler, epoch_index, tb_writer, device):
    model.train()
    running_loss = 0.
    print(f'epoch : {epoch_index}')
    tb_writer.add_scalar("LR/train", optimizer.param_groups[0]['lr'], epoch_index)
    # print(optimizer.param_groups[0]['lr'])
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for data in tqdm(trainloader):
        fmaps = []
        # Every data instance is an input + label pair
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    # learning rate 조정
    scheduler.step()
    tb_writer.add_scalar("Loss/train", running_loss, epoch_index)

    return running_loss


def evaluate(model, testloader, device, epoch_index, tb_writer, model_name):
    model.eval()
    correct = 0
    total = 0
    concat_outputs = []
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            concat_outputs.append(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Accuracy : {acc}%')
    tb_writer.add_scalar(f"Acc/test - {model_name}", acc, epoch_index)
    concat_outputs = torch.cat(concat_outputs, 0)

    return acc, concat_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tb', type=str, default='test')
    parser.add_argument('-g', '--gpu', type=str, default='cuda:0')
    parser.add_argument('-e', '--epochs', type=int, default=90)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-seed', type=int, default=77)
    parser.add_argument('-t', '--temperature', type=int, default=30)
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('-d_dir', type=str, default='../datasets/')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    writer = SummaryWriter(f'runs/{args.tb}')

    for name, value in vars(args).items():
        print(f'{name} : {value}')
        writer.add_text(f'{name}', f'{value}')

    reproduce(args.seed)  # setting random seed





