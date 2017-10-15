import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('../vanilla_densenet_small/')
from densenet import DenseNet


def get_model(learning_rate=1e-3):

    model = DenseNet(
        growth_rate=12, block_config=(8, 12, 10),
        num_init_features=48, bn_size=4, drop_rate=0.25,
        final_drop_rate=0.25, num_classes=200
    )

    # set the first layer not trainable
    model.features.conv0.weight.requires_grad = False

    # the last fc layer
    weights = [
        p for n, p in model.named_parameters()
        if 'classifier.weight' in n
    ]
    biases = [model.classifier.bias]
    
    # all conv layers except the first
    weights_to_be_quantized = [
        p for n, p in model.named_parameters()
        if 'conv' in n and ('dense' in n or 'transition' in n)
    ]
    
    # parameters of batch_norm layers
    bn_weights = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'bias' in n
    ]

    params = [
        {'params': weights, 'weight_decay': 1e-4},
        {'params': weights_to_be_quantized},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.Adam(params, lr=learning_rate)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer
