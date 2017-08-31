import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append('../utils/')
from train_utils import _evaluate, _accuracy, _is_early_stopping


def initial_scales(kernel):
    return 1.0, 1.0


def quantize(kernel, w_p, w_n, t):
    delta = t*kernel.abs().max()
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    return w_p*a + (-w_n*b)


def get_grads(kernel_grad, kernel, w_p, w_n, t):
    delta = t*kernel.abs().max()
    # masks
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    c = torch.ones(kernel.size()).cuda() - a - b
    # scaled kernel grad and grad for scaling factors (w_p, w_n)
    return w_p*a*kernel_grad + w_n*b*kernel_grad + 1.0*c*kernel_grad,\
        (a*kernel_grad).sum(), (b*kernel_grad).sum()


def _optimization_step(model, criterion, optimizer_list, t,
                       x_batch, y_batch):
    
    # optimizers for
    # full model (all weights including quantized weights), 
    # backup of full precision weights,
    # scaling factors for each layer
    optimizer, optimizer_fp, optimizer_sf = optimizer_list
    
    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
    # use quantized model
    logits = model(x_batch)

    # compute logloss
    loss = criterion(logits, y_batch)
    batch_loss = loss.data[0]

    # compute accuracies
    pred = F.softmax(logits)
    batch_accuracy, batch_top5_accuracy = _accuracy(y_batch, pred, top_k=(1, 5))

    optimizer.zero_grad()
    optimizer_fp.zero_grad()
    optimizer_sf.zero_grad()
    # compute grads for quantized model
    loss.backward()
    
    # all quantized kernels
    all_kernels = optimizer.param_groups[1]['params']
    # their full precision backups
    all_fp_kernels = optimizer_fp.param_groups[0]['params']
    # two scaling factors for each quantized kernel
    scaling_factors = optimizer_sf.param_groups[0]['params']

    for i in range(len(all_kernels)):
        
        # get quantized kernel
        k = all_kernels[i]
        
        # get corresponding full precision kernel
        k_fp = all_fp_kernels[i]
        
        # get scaling factors for the quantized kernel
        f = scaling_factors[i]
        w_p, w_n = f.data[0], f.data[1]
        
        # get modified grads
        k_fp_grad, w_p_grad, w_n_grad = get_grads(k.grad.data, k.data, w_p, w_n, t)

        # grad for full precision kernel
        k_fp.grad = Variable(k_fp_grad)
        
        # we don't need to update the quantized kernel directly
        k.grad.data.zero_()
        
        # grad for scaling factors
        f.grad = Variable(torch.FloatTensor([w_p_grad, w_n_grad]).cuda())
    
    # update the last fc layer and all batch norm params in quantized model
    optimizer.step()
    
    # update full precision kernels
    optimizer_fp.step()
    
    # update scaling factors
    optimizer_sf.step()
    
    # update quantized kernels
    for i in range(len(all_kernels)):
        
        k = all_kernels[i]
        k_fp = all_fp_kernels[i]
        f = scaling_factors[i]
        w_p, w_n = f.data[0], f.data[1]
        
        k.data = quantize(k_fp.data, w_p, w_n, t)
    
    return batch_loss, batch_accuracy, batch_top5_accuracy


# just training helper, nothing special
def train(model, criterion, optimizer_list, t,
          train_iterator, n_epochs, steps_per_epoch,
          val_iterator, n_validation_batches,
          patience=10, threshold=0.01, lr_scheduler=None):
    
    # collect losses and accuracies here
    all_losses = []
    
    is_reduce_on_plateau = isinstance(lr_scheduler, ReduceLROnPlateau)
    
    running_loss = 0.0
    running_accuracy = 0.0
    running_top5_accuracy = 0.0
    start_time = time.time()
    model.train()

    for epoch in range(0, n_epochs):
        
        # main training loop
        for step, (x_batch, y_batch) in enumerate(train_iterator, 1 + epoch*steps_per_epoch):

            batch_loss, batch_accuracy, batch_top5_accuracy = _optimization_step(
                model, criterion, optimizer_list, t, x_batch, y_batch
            )
            running_loss += batch_loss
            running_accuracy += batch_accuracy
            running_top5_accuracy += batch_top5_accuracy
        
        # evaluation
        model.eval()
        test_loss, test_accuracy, test_top5_accuracy = _evaluate(
            model, criterion, val_iterator, n_validation_batches
        )
        
        # collect evaluation information and print it
        all_losses += [(
            epoch,
            running_loss/steps_per_epoch, test_loss,
            running_accuracy/steps_per_epoch, test_accuracy,
            running_top5_accuracy/steps_per_epoch, test_top5_accuracy
        )]
        print('{0}  {1:.3f} {2:.3f}  {3:.3f} {4:.3f}  {5:.3f} {6:.3f}  {7:.3f}'.format(
            *all_losses[-1], time.time() - start_time
        ))
        
        # it watches test accuracy
        # and if accuracy isn't improving then training stops
        if _is_early_stopping(all_losses, patience, threshold):
            print('early stopping!')
            break
        
        if lr_scheduler is not None:
            # change learning rate
            if not is_reduce_on_plateau:
                lr_scheduler.step()
            else:
                lr_scheduler.step(test_accuracy)
                
        running_loss = 0.0
        running_accuracy = 0.0
        running_top5_accuracy = 0.0
        start_time = time.time()
        model.train()

    return all_losses
