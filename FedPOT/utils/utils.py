import time
import numpy as np
import torch
import random
from PIL import Image
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision.models as models
from scipy.ndimage.interpolation import rotate as scipyrotate
from torchvision.models import vgg16
from copy import deepcopy
from random import choice
from tqdm import tqdm
from sklearn.utils import shuffle
import torchvision.datasets as datasets
import csv
import os
import math

def load_users(args, trains, labels1, labels2, data_size, targets, datasets, num_classes):
    if trains:

        if args.distribution in ["class-uniform", "quantity-skewed"]:
            all_index = []

            for l in range(num_classes):
                select_indexes = list(np.where(np.array(targets) == l)[0])

                index = random.sample(select_indexes, int(data_size // num_classes))

                all_index.extend(index)

            datas = Subset(datasets, all_index)
        else:

            all_index = []
            for l in [labels1, labels2]:
                select_indexes = list(np.where(np.array(targets) == l)[0])

                index = random.sample(select_indexes, int(data_size * 0.4))

                all_index.extend(index)

            counts = []
            while True:
                index = random.randint(0, len(targets) - 1)

                if (index not in counts) and (targets[index] not in [labels1, labels2]):
                    all_index.append(index)

                if len(all_index) == data_size:
                    break

            datas = Subset(datasets, all_index)
    else:
        lens = len(datasets)
        all_index = range(lens)

        if labels1 == 0:
            all_index = all_index[:lens // 2]
        else:
            all_index = all_index[lens // 2:]

        datas = Subset(datasets, all_index)

        return datas

    return datas

def get_test_dataset(dataset, iftest, data_path, args):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        dst_test_MNIST = datasets.MNIST(data_path, train=False, download=True, transform=transform)

        if iftest:
            dst_test = load_users(args, False, 1, 0, 0, None, dst_test_MNIST, num_classes)
        else:
            dst_test = load_users(args, False, 0, 0, 0, None, dst_test_MNIST, num_classes)

    else:
        exit('unknown dataset: %s' % dataset)

    return channel, num_classes, dst_test

def get_dataset(size_list, dataset, users, data_path, args):

    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        dst_train_MNIST = datasets.MNIST(data_path, train=True, transform=transform)

        targets = []
        for (img, tar) in dst_train_MNIST:
            targets.append(tar)

        if args.distribution in ["class-uniform", "quantity-skewed"]:

            whole_train = []

            for i in range(users):
                data_users = load_users(args, True, 0, 0, size_list[i], targets, dst_train_MNIST, num_classes)
                whole_train.append([data_users, size_list[i]])

        else:
            print("class-skewed")

            whole_train = []

            for i in range(args.users):
                labels1 = random.randint(0, num_classes - 1)
                labels2 = random.randint(0, num_classes - 1)

                while labels1 == labels2:
                    labels1 = random.randint(0, num_classes - 1)
                    labels2 = random.randint(0, num_classes - 1)

                print(labels1, labels2)
                data_users = load_users(args, True, labels1, labels2, size_list[i], targets, dst_train_MNIST, num_classes)

                whole_train.append([data_users, size_list[i]])

    else:
        exit('unknown dataset: %s'% dataset)


    return channel, im_size, num_classes, whole_train


def eval_net(model_state, evalloader):
    criterion = nn.CrossEntropyLoss().cuda()
    net = vgg16()
    net.eval()

    net.load_state_dict(model_state)

    loss_real, acc_real = epoch('eval', evalloader, net, None, criterion)

    return loss_real, acc_real

# fedavg
def fedavg(size_list, local_model_list):
    next_state = OrderedDict()
    whole_sizes = 0
    users = len(size_list)
    for i in range(users):
        whole_sizes += size_list[i]

    indexes = 0
    for i in range(users):
        local_states = local_model_list[i].state_dict()
        for key in local_states.keys():
            if indexes == 0:
                next_state[key] = (local_states[key] * size_list[i] / float(whole_sizes))
            else:
                next_state[key] += (local_states[key] * size_list[i] / float(whole_sizes))
        indexes += 1

    return next_state


def get_agg(size_list, local_model_list):
    server_state = fedavg(size_list, local_model_list)

    return server_state


def fl(args, channel, num_classes, dataset, server_state):
    criterion = nn.CrossEntropyLoss().cuda()

    net = vgg16()
    
    model_list = []

    for i in range(args.users):
        net.load_state_dict(server_state)

        net.train()
        
        optimizer_net = torch.optim.Adam(net.parameters(), lr=args.lr_net)
        optimizer_net.zero_grad()

        for inol in range(args.inner_loops):
            net.train()
            epoch('train', dataset[i], net, optimizer_net, criterion)
        net.eval()
        
        model_list.append(net)
    
    return model_list


def epoch(mode, dataloader, net, optimizer, criterion):
    loss_avg, acc_avg, num_exp = 0, 0, 0

    net = net.cuda()
    criterion = criterion.cuda()

    if mode == 'train':
        net.train()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            img = batch[0].cuda()
            lab = batch[1].long().cuda()
            n_b = lab.shape[0]

            output = net(img)
            loss = criterion(output, lab)

            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item() * n_b
            acc_avg += acc
            num_exp += n_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    else:
        net.eval()

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                net.eval()

                img = batch[0].cuda()
                lab = batch[1].long().cuda()
                n_b = lab.shape[0]

                output = net(img)
                loss = criterion(output, lab)

                acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

                loss_avg += loss.item() * n_b
                acc_avg += acc
                num_exp += n_b

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg
