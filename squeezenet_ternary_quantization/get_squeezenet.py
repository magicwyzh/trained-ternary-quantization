import torch.nn as nn
import torch.optim as optim
from torch.nn.init import constant

import sys
sys.path.append('../vanilla_squeezenet/')
from squeezenet import SqueezeNet


def get_model():

    model = SqueezeNet()

    # create different parameter groups
    weights = [
        (n, p) for n, p in model.named_parameters()
        if 'weight' in n and not 'bn' in n and not 'features.1.' in n
    ]
    
    weights_to_be_quantized = [
        p for n, p in weights
        if not ('classifier' in n or 'features.0.' in n)
    ]
    weights = [
        p for n, p in weights
        if 'classifier' in n or 'features.0.' in n
    ]
    
    biases = [model.classifier[1].bias]
    bn_weights = [
        p for n, p in model.named_parameters()
        if ('bn' in n or 'features.1.' in n) and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if ('bn' in n or 'features.1.' in n) and 'bias' in n
    ]
    
    for p in bn_weights:
        constant(p, 1.0)
    for p in bn_biases:
        constant(p, 0.0)

    params = [
        {'params': weights, 'weight_decay': 1e-5},
        {'params': weights_to_be_quantized},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.SGD(params, lr=1e-4, momentum=0.9, nesterov=True)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
