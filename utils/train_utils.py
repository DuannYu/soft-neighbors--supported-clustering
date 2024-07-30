"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
import torch.nn.functional as F

from utils.utils import AverageMeter, ProgressMeter


def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, (batch, _) in enumerate(train_loader):

        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        # import pdb; pdb.set_trace()
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
        anchor_augmented = batch['anchor_augmented'].to('cuda',non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
                anchor_augmented_features = model(anchor_augmented, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')
            anchor_augmented_output = model(anchor_augmented_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_features = model(anchors, forward_pass='backbone')
            neighbors_features = model(neighbors, forward_pass='backbone')
            anchor_augmented_features = model(anchor_augmented, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')
            anchor_augmented_output = model(anchor_augmented_features, forward_pass='head')

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead, anchor_augmented_output_subhead in zip(anchors_output, neighbors_output, anchor_augmented_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_features,
                                                                         anchor_augmented_features,
                                                                         anchors_output_subhead,
                                                                         neighbors_output_subhead, 
                                                                         anchor_augmented_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, (batch, _) in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]
        # import pdb; pdb.set_trace()
        loss = criterion(output, output_augmented)
        losses.update(loss.item())
        # import pdb; pdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)

def boosting_train(train_loader, model, criterion_ins, criterion_clu, pseudo_labels, optimizer, epoch, ema=None):
    """ 
    Boosting-Train based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, (batch, index) in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            features = model(images, forward_pass='backbone')
            output = model(features, forward_pass='head')[0]
            # output = F.softmax(output, dim=1)
            pseudo_labels_cur, index_cur = criterion_ins.generate_pseudo_labels(
                output, pseudo_labels[index].to(output.device), index.to(output.device)
            )
            pseudo_labels[index_cur] = pseudo_labels_cur
            
        output_augmented = model(images_augmented)[0]
        # output_augmented = F.softmax(output_augmented, dim=1)
        # features = F.normalize(features,dim=1)
        # features_augmented = F.normalize(features_augmented,dim=1)
        
        loss_ins = 0.
        loss_ins = criterion_ins(output, output_augmented, output, output_augmented, pseudo_labels[index].to(output.device))
        # import pdb; pdb.set_trace()
        # loss_ins = criterion_ins(features, features_augmented, output, output_augmented, pseudo_labels[index].to(output.device))
        # loss_clu = criterion_clu(output_augmented, pseudo_labels[index].to(output.device))
        loss_clu = criterion_clu(epoch, output, output_augmented)
        # loss_clu = 0.
        loss = loss_ins + loss_clu
        # import pdb; pdb.set_trace()
            
        losses.update(loss.item())
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)
            
    return pseudo_labels