import torch.nn as nn
import torch.optim as optim
from torch.nn.init import constant, kaiming_uniform
from densenet import DenseNet


def get_model():

    model = DenseNet(
        growth_rate=24, block_config=(3, 6, 8, 6),
        num_init_features=64, bn_size=4, drop_rate=0.25,
        final_drop_rate=0.25, num_classes=200
    )

    # create different parameter groups
    weights = [
        p for n, p in model.named_parameters()
        if 'conv' in n or 'classifier.weight' in n
    ]
    biases = [model.classifier.bias]
    bn_weights = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'bias' in n
    ]

    # parameter initialization
    for p in weights:
        kaiming_uniform(p)
    for p in biases:
        constant(p, 0.0)
    for p in bn_weights:
        constant(p, 1.0)
    for p in bn_biases:
        constant(p, 0.0)

    params = [
        {'params': weights, 'weight_decay': 1e-3},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.SGD(params, lr=1e-1, momentum=0.9, nesterov=True)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
