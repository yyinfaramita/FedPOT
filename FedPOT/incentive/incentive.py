import time
import numpy as np
import torch
import torch.nn as nn
from utils.utils import epoch, get_agg
from torchvision.models import vgg16
from copy import deepcopy
import math
import os
import random
from itertools import combinations
from torch.linalg import norm
import torch.nn.functional as F


# train under-train model
def train_fraud_attacker(args, channel, num_classes, ori_state, ori_dataset):
    criterion = torch.nn.CrossEntropyLoss().cuda()

    max_net = vgg16()
    max_net.train()
    max_net.load_state_dict(ori_state)

    optimizer_net_temp = torch.optim.Adam(max_net.parameters(), lr=args.attack_lr)
    optimizer_net_temp.zero_grad()

    for i in range(args.attack_inner_loops):
        max_net.train()

        epoch('train', ori_dataset, max_net, optimizer_net_temp, criterion)

    return max_net

# incentive mechanism
def get_reward(args, server_state, size_list, evalloader, testloader, channel, num_classes, local_model_list):

    criterion = nn.CrossEntropyLoss().cuda()
    rewards_list = []

    default_net = vgg16()
    default_net.eval()

    aggregation_state = get_agg(args, size_list, server_state, channel, num_classes, local_model_list,
                                evalloader, testloader)

    default_net.load_state_dict(aggregation_state)
    loss_default, acc_default = epoch('eval', evalloader, default_net, None, criterion)

    for j in range(args.users):
        selected_model_list = []
        selected_size_list = []
        for i in range(args.users):
            if i != j:
                selected_model_list.append(local_model_list[i])
                selected_size_list.append(size_list[i])

        selected_state = get_agg(args, selected_size_list, server_state, channel, num_classes, selected_model_list,
                                 evalloader, testloader)

        default_net.load_state_dict(selected_state)
        loss_selected, acc_selected = epoch('eval', evalloader, default_net, None, criterion)

        rewards_list.append(- (loss_default - loss_selected))

    return rewards_list

