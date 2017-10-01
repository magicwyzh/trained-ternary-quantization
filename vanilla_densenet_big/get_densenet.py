import torch.nn as nn
import torch.optim as optim
from torch.nn.init import constant, kaiming_uniform
from densenet import DenseNet
import torch.utils.model_zoo as model_zoo


def get_model():

    # densenet-121
    model = DenseNet(
        num_init_features=64, growth_rate=32,
        block_config=(6, 12, 24, 16),
        drop_rate=0, final_drop_rate=0.2
    )

    state_dict = model_zoo.load_url(
        'https://download.pytorch.org/models/densenet121-241335ed.pth'
    )

    # resize


    model.load_state_dict()
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

    # parameter initializations
    for p in weights:
        kaiming_uniform(p)
    for p in biases:
        constant(p, 0.0)
    for p in bn_weights:
        constant(p, 1.0)
    for p in bn_biases:
        constant(p, 0.0)

    params = [
        {'params': weights, 'weight_decay': 1e-4},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.SGD(params, lr=1e-1, momentum=0.95, nesterov=True)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer
