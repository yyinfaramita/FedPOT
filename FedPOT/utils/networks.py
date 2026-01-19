import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
import copy
import math
from utils.bert_layer import BertConfig, BERTEmbeddings, BERTLayer, BERTPooler, BERTLayerNorm
import os
from six import add_metaclass
from contextlib import contextmanager
import torchvision


''' VGG16 '''
class VGG16(nn.Module):
    def __init__(self, channel, num_classes):
        super(VGG16, self).__init__()
        self.channel = channel
        self.vgg = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        features_layers, features2_layers, features3_layers, features4_layers, features5_layers = [], [], [], [], []
        in_channels = channel
        temp_features = []
        for i in range(len(self.vgg)):
            x = self.vgg[i]
            temp_features += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if in_channels==1 else 1),
                    nn.ReLU(inplace=True)]
            if i in [1, 3, 6, 9, 12]:
                temp_features += [nn.MaxPool2d(kernel_size=2, stride=2)]
            if i == 12:
                temp_features += [nn.AvgPool2d(kernel_size=1, stride=1)]

            if i == 1:
                features_layers = copy.deepcopy(temp_features)
                temp_features = []
            elif i == 3:
                features2_layers = copy.deepcopy(temp_features)
                temp_features = []
            elif i == 6:
                features3_layers = copy.deepcopy(temp_features)
                temp_features = []
            elif i == 9:
                features4_layers = copy.deepcopy(temp_features)
                temp_features = []
            elif i == 12:
                features5_layers = copy.deepcopy(temp_features)
                temp_features = []

            in_channels = x

        self.features1 = nn.Sequential(*features_layers)
        self.features2 = nn.Sequential(*features2_layers)
        self.features3 = nn.Sequential(*features3_layers)
        self.features4 = nn.Sequential(*features4_layers)
        self.features5 = nn.Sequential(*features5_layers)

        self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, is_feat=False, preact=False):
        x1 = self.features1(x)

        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x6 = x5.view(x.size(0), -1)
        x7 = self.classifier(x6)

        if is_feat:
            if preact:
                return [x1, x2, x3, x4, x5], F.log_softmax(x7, dim=1)
            else:
                return [x1, x2, x3, x4, x5], F.log_softmax(x7, dim=1)
        else:
            return x7

