# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial
from torchsummary import summary
from torchvision import models


def resnet50(pretrained, num_classes):
    resnet50_pretrained = models.resnet50(pretrained=pretrained)
    num_ftrs = resnet50_pretrained.fc.in_features
    resnet50_pretrained.fc = nn.Linear(num_ftrs, num_classes)

    return resnet50_pretrained

if __name__ == "__main__":
    model = resnet50(pretrained=True, num_classes=10).cuda()
    summary(model, (3, 224, 224), device='cuda')
