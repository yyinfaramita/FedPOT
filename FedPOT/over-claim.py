import os
import random
import argparse
import numpy as np
import torch
from utils.utils import fl, get_dataset, eval_net, get_test_dataset, get_agg
from torchvision.models import vgg16
from incentive.incentive import get_reward, train_fraud_attacker
import matplotlib
import copy
import math
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.nn as nn


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.gpu

    channel, num_classes, dst_test = get_test_dataset(args.dataset, True, args.data_path, args)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_eval)

    channel, num_classes, dst_val = get_test_dataset(args.dataset, False, args.data_path, args)
    valloader = torch.utils.data.DataLoader(dst_val, batch_size=args.batch_eval)

    net = vgg16()
    net.train()
    start_state = net.state_dict()

    _, acc_ori = eval_net(start_state, testloader)
    print("Original Server Accuracy: %.4f" % acc_ori)

    # data distribution
    size_distribution = [1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3]
    size_list = []
    for i in range(args.users):
        if args.distribution == "class-uniform":
            size_list.append(args.train_size)
        elif args.distribution == "class-skewed":
            size_list.append(args.train_size)
        elif args.distribution == "quantity-skewed":
            size_list.append(int(size_distribution[i] / 2 * args.train_size))

    channel, im_size, num_classes, whole_train = get_dataset(size_list, args.dataset, args.users, args.data_path, args)

    realloader_list = []
    for i in range(args.users):
        trainloader = torch.utils.data.DataLoader(whole_train[i][0], batch_size=args.batch_train, shuffle=True)

        realloader_list.append(trainloader)
    ####################################################################################################################
    communication_loops = 1
    last_server_state = deepcopy(start_state)

    per_flops = 1 # FLOPs per one sample of one epoch, decided by model structure

    while True:
        print("Iteration = %d, data size = %.2f/%.2f" % (communication_loops, args.train_size, args.claim_size))

        local_model_list = fl(args, channel, num_classes, realloader_list, testloader, last_server_state)

        server_honest_state = get_agg(args, size_list, local_model_list)

        _, acc_real = eval_net(server_honest_state, testloader)
        print("Global Accuracy when Honest = %.4f" % (acc_real))

        honest_rewards_list = get_reward(args, last_server_state,
                                       size_list, valloader, testloader, realloader_list, 
                                       channel, num_classes, local_model_list)
        
        real_wholes = 0.0
        for i in honest_rewards_list:
            if i > 0:
                real_wholes += i
                
        whole_fraud_rewards_list = [0] * args.users

        for f in range(args.users):
            real_cost = size_list[f] * per_flops * args.inner_loops

            claim_size_list = deepcopy(size_list)
            claim_size_list[f] = int(args.claim_size * size_list[f])

            honest_rewards = 0
            if honest_rewards_list[f] > 0:
                honest_rewards = honest_rewards_list[f] / real_wholes

            fraud_rewards = 0

            fraud_rewards_list = get_reward(args, last_server_state, claim_size_list,
                                            valloader, testloader, channel,
                                            num_classes, local_model_list)

            whole_fraud = 0.0
            for j in fraud_rewards_list:
                if j > 0:
                    whole_fraud += j

            if fraud_rewards_list[f] > 0:
                fraud_rewards = fraud_rewards_list[f] / whole_fraud

            fraud_reward_cost_ratio = (fraud_rewards * 100 / real_cost - honest_rewards * 100 / real_cost) * pow(10, 12)

            whole_fraud_rewards_list[f] = fraud_reward_cost_ratio

        indexed_data = list(enumerate(whole_fraud_rewards_list))
        sorted_data = sorted(indexed_data, reverse=True, key=lambda x: x[1])
        sorted_elements = [index for index, element in sorted_data]
        attackers_list = sorted_elements[:args.fusers]

        claim_size_list = deepcopy(size_list)
        for attackers in attackers_list:
            claim_size_list[attackers] = int(args.claim_size * size_list[attackers])

        server_fraud_state, ex_pa = get_agg(args, claim_size_list, local_model_list)

        _, acc_fraud = eval_net(server_fraud_state, testloader)
        print("Global Accuracy after RFA = %.4f" % (acc_fraud))
        
        fraud_rewards_list = get_reward(args, last_server_state, claim_size_list,
                                        valloader, testloader, channel, num_classes, local_model_list)

        whole_fraud = 0
        for j in fraud_rewards_list:
            if j > 0:
                whole_fraud += j

        for i in range(args.users):
            real_cost = size_list[i] * per_flops * args.inner_loops

            fraud_rewards = 0
            if fraud_rewards_list[i] > 0:
                fraud_rewards = fraud_rewards_list[i] / whole_fraud

            honest_rewards = 0
            if honest_rewards_list[i] > 0:
                honest_rewards = honest_rewards_list[i] / real_wholes

            rcr_increase = ((fraud_rewards * 100) / real_cost - (
                    honest_rewards * 100) / real_cost) * pow(10, 12)

            print("%d, Real Rewards = %.2f, Fraud Rewards = %.2f, RCR Increase = %.2f, " %
                (i, honest_rewards, fraud_rewards, rcr_increase))

        #################################################################################
        # Update the global model after RFA
        # last_server_state = deepcopy(server_fraud_state)
        #################################################################################
        # Update the global model when all honest
        last_server_state = deepcopy(server_honest_state)
        
        if communication_loops == args.global_loops:
            break

        communication_loops += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--gpu', type=str, default="0", help='number of the gpu device')

    parser.add_argument('--reward', type=str, default='refiner', help='incentive mechanism')

    parser.add_argument('--global_loops', type=int, default=65, help='number of global loops')
    parser.add_argument('--inner_loops', type=int, default=5, help='local update loop')

    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--model', type=str, default='VGG16', help='model')

    parser.add_argument('--distribution', type=str, default='same-same', help='data distribution '
                        '(class-uniform, class-skewed, quantity-skewed')

    parser.add_argument('--aggregation', type=str, default='fedavg', help='aggregation method')
    parser.add_argument('--defense', type=str, default='none', help='defense method')

    parser.add_argument('--train_size', type=int, default=5000, help='true data set size')
    parser.add_argument('--claim_size', type=int, default=5000, help='claim data set size')

    parser.add_argument('--users', type=int, default=10, help='number of users')
    parser.add_argument('--fusers', type=int, default=1, help='number of attackers')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    parser.add_argument('--batch_train', type=int, default=50, help='batch size for training networks')
    parser.add_argument('--batch_eval', type=int, default=50, help='batch size for training networks')

    parser.add_argument('--seed', type=int, default=2026, help='local update loop')

    args = parser.parse_args()

    main(deepcopy(args))







