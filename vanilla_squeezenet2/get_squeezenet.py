import torch.nn as nn
import torch.optim as optim
from torch.nn.init import constant
from squeezenet import SqueezeNet


def get_model():

    model = SqueezeNet()

    # create different parameter groups
    weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and len(p.size()) == 4
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
    prelu = [
        p for n, p in model.named_parameters() if 'activation' in n
    ]
    prelu += [model.features[2].weight, model.classifier[2].weight]
    
    for p in bn_weights:
        constant(p, 1.0)
    for p in bn_biases:
        constant(p, 0.0)

    params = [
        {'params': weights, 'weight_decay': 1e-4},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases},
        {'params': prelu}
    ]
    optimizer = optim.SGD(params, lr=4e-2, momentum=0.9, nesterov=True)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
