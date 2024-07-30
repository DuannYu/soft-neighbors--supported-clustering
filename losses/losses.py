"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

EPS=1e-8

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        # if not (mask != 0).any():
        #     raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        import pdb; pdb.set_trace()
        mask = max_prob > self.threshold 
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        return loss


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

def get_radius(features1, features2, topk=1):
    assert topk <= features1.shape[0], 'topk={} should less than # of features={}'.format(topk, features1.shape[0])

    b, n = features1.shape
    # compute topk distance as radius
    pairwise_distance = torch.cdist(features1, features2)
    values, _ = torch.sort(pairwise_distance)
    radius = values[:, topk].repeat(b,1).t()

    return radius, pairwise_distance
     
def soft_loss(features, features_prob, neighbors=None, topk=1):
    """
    Helper function to compute the soft bce loss over the batch 

    input: 
        - data: logits for anchor images w/ shape [b, num_classes]
    output: 
        - softentropy value [is ideally -log(num_classes)]
    """   
    assert topk <= features.shape[0], 'topk={} should less than # of features={}'.format(topk, features.shape[0])

    b, n = features.shape
    # compute topk distance as radius
    radius, pairwise_distance = get_radius(features, features, topk=1)

    # compute weights
    weights = (pairwise_distance - radius) / radius
    clip_weights = 1 - torch.clip(weights, 0, 1) - torch.eye(features.shape[0]).cuda()

    # remove radius-made pointss
    _, indices = torch.max(clip_weights, dim=1, keepdim=True)
    clip_weights = torch.scatter(clip_weights, 1, indices, 0)

    mask = torch.sign(clip_weights)
    # import pdb; pdb.set_trace()
    similarity = torch.mm(features_prob, features_prob.t())
    local_loss = clip_weights * similarity + mask*(1-clip_weights)*(1-similarity)

    global_loss = 0
    if not neighbors == None:
        # compute topk distance as radius
        radius, pairwise_distance = get_radius(features, neighbors, topk=1)

        # compute weights
        weights = (pairwise_distance - radius) / radius
        clip_weights = 1 - torch.clip(weights, 0, 1) - torch.eye(features.shape[0]).cuda()

        # remove radius-made pointss
        _, indices = torch.max(clip_weights, dim=1, keepdim=True)
        clip_weights = torch.scatter(clip_weights, 1, indices, 0)

        mask = torch.sign(clip_weights)
        # import pdb; pdb.set_trace()
        similarity = torch.mm(features_prob, features_prob.t())
        global_loss = clip_weights * similarity + mask*(1-clip_weights)*(1-similarity)

    alpha = 0.0; beta = 1.0
    loss = alpha*local_loss + beta*global_loss
    # log tricks
    # loss = -clip_weights * pairwise_distance + mask*(1-clip_weights)*torch.log(1-torch.exp(-pairwise_distance)+1e-8)
    return -loss.mean()



def soft_neighbors_loss(anchors, neighbors, anchors_prob, positives_prob):
    '''
    L1使用加权方法
    '''
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        # compute topk distance as radius
        radius, pairwise_distance = get_radius(anchors, neighbors, topk=1)

        # W_P compute weights ref: 2023 ICLR
        weights = 1-torch.clamp((pairwise_distance-radius)/radius, 0, 1) - torch.eye(len(anchors)).cuda()

        top, indices = torch.topk(weights, 1)
        res = torch.zeros(len(anchors), len(anchors)).cuda()
        weights = res.scatter(1, indices, top)

        # mask = torch.eye(len(s_emb), len(s_emb)).byte().cuda()
        # W_C.masked_fill_(mask, 1)
        # import pdb; pdb.set_trace()        
        pos_mask = torch.sign(weights).cuda()
        neg_mask = 1 - pos_mask - torch.eye(len(anchors)).cuda()

    inner_prod = torch.mm(anchors_prob, positives_prob.t())
    exp_inner_prod = torch.exp(inner_prod)

    pull_loss = weights*torch.log(exp_inner_prod)
    # push_loss = torch.sum(neg_mask*(1-weights)*log_inner_prod)

    loss = pull_loss
    return -loss.mean()

def soft_neighbors_loss_L2(anchors, neighbors, anchors_prob, positives_prob, global_neighbors=False, boosting=False):
    '''
    对应手稿L2
    '''
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        # compute topk distance as radius
        radius, pairwise_distance = get_radius(anchors, neighbors, topk=1)
        # import pdb; pdb.set_trace()
        # W_P compute weights ref: 2023 ICLR
        if global_neighbors:
            weights = 1-torch.clamp((pairwise_distance-radius)/radius, 0, 1)
        else:
            weights = 1-torch.clamp((pairwise_distance-radius)/radius, 0, 1) - torch.eye(len(anchors)).cuda()
            
        top, indices = torch.topk(weights, 10)
        res = torch.zeros(len(anchors), len(anchors)).cuda()
        weights = res.scatter(1, indices, top)

        # mask = torch.eye(len(s_emb), len(s_emb)).byte().cuda()
        # W_C.masked_fill_(mask, 1)      
        mask = torch.sign(weights).cuda()

    inner_prod = torch.mm(anchors_prob, positives_prob.t())

    pull_loss = weights*inner_prod
    push_loss = (1-weights)*(1-inner_prod)

    
    if boosting:
        return pull_loss
    else:
        loss = pull_loss + push_loss
        return loss.mean()

def soft_neighbors_loss_L3(anchors, neighbors, anchors_prob, positives_prob):
    '''
    对应手稿L3 
    '''
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        # compute topk distance as radius
        radius, pairwise_distance = get_radius(anchors, neighbors, topk=1)

        # W_P compute weights ref: 2023 ICLR
        weights = 1-torch.clamp((pairwise_distance-radius)/radius, 0, 1) - torch.eye(len(anchors)).cuda()

        top, indices = torch.topk(weights, 5)
        res = torch.zeros(len(anchors), len(anchors)).cuda()
        weights = res.scatter(1, indices, top)

        # mask = torch.eye(len(s_emb), len(s_emb)).byte().cuda()
        # W_C.masked_fill_(mask, 1)      
        mask = torch.sign(weights).cuda()

    inner_prod = torch.mm(anchors_prob, positives_prob.t())
    exp_inner_prod = torch.log(inner_prod)

    pull_loss = weights*inner_prod
    push_loss = mask*(1-weights)*torch.log(1-exp_inner_prod)
    # import pdb; pdb.set_trace()
    loss = pull_loss + push_loss
    return -loss.mean()


def soft_neighbors_loss2(anchors, neighbors, anchors_prob, positives_prob):

    with torch.no_grad():
        # compute topk distance as radius
        radius, pairwise_distance = get_radius(anchors, neighbors, topk=1)

        # W_P compute weights ref: 2023 ICLR
        weights = 1-torch.clamp((pairwise_distance-radius)/radius, 0, 1) - torch.eye(len(anchors)).cuda()

        top, indices = torch.topk(weights, 10)
        res = torch.zeros(len(anchors), len(anchors)).cuda()
        weights = res.scatter(1, indices, top)

        # mask = torch.eye(len(s_emb), len(s_emb)).byte().cuda()
        # W_C.masked_fill_(mask, 1)
                    
        pos_mask = torch.sign(weights).cuda()
        neg_mask = 1 - pos_mask - torch.eye(len(anchors)).cuda()
        
        inner_prod = torch.mm(anchors_prob, positives_prob.t())
        exp_inner_prod = torch.exp(inner_prod)

    pull_loss = torch.sum(weights*exp_inner_prod, axis=1)
    push_loss = torch.sum(neg_mask*exp_inner_prod)

    unweight_pull_loss = torch.sum(pos_mask*exp_inner_prod)
    # import pdb; pdb.set_trace()
    loss = torch.log(pull_loss / (unweight_pull_loss + push_loss))

    return -loss.mean()

def column_ce_loss(anchors_prob, positives_prob):
    _, n = anchors_prob.size()
    anchors_prob_c = F.normalize(anchors_prob, dim = 0)
    positives_prob_c = F.normalize(positives_prob, dim = 0)

    similarity = torch.mm(anchors_prob_c.t(), positives_prob_c)
    labels = torch.tensor(list(range(n))).cuda()
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(similarity, labels)

    return ce_loss

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0
    
    def forward(self, anchors_features, augments_features, anchors, neighbors, augments):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
        augments_prob = self.softmax(augments)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # soft pairwise loss
        # pairwise_loss = soft_loss(anchors, anchors_prob, neighbors)
        # soft_loss = soft_neighbors_loss_L2(anchors, anchors, anchors_prob, anchors_prob)
        # import pdb; pdb.set_trace()
        soft_loss = soft_neighbors_loss_L2(anchors_features, anchors_features, anchors_prob, anchors_prob)

        # column ce loss
        ce_loss = column_ce_loss(anchors_prob, positives_prob)
        # import pdb; pdb.set_trace()

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)
        # import pdb; pdb.set_trace()

        # Global Loss
        global_loss = 0
        global_aug = True
        if global_aug == True:
            similarity = torch.bmm(anchors_prob.view(b, 1, n), augments_prob.view(b, n, 1)).squeeze()
            ones = torch.ones_like(similarity)
            # bce_aug = self.bce(similarity, ones)
            soft_aug = soft_neighbors_loss_L2(anchors_features, augments_features, anchors_prob, augments_prob,global_neighbors=global_aug)

            # ce_aug = column_ce_loss(anchors_prob, augments_prob)

            global_loss = soft_aug
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # kl_loss = F.kl_div(anchors_prob.log(), boost_prob)
        # import pdb; pdb.set_trace()
        # Total loss
        # total_loss = consistency_loss - self.entropy_weight*entropy_loss + soft_loss + ce_loss + global_loss
        total_loss = consistency_loss - self.entropy_weight*entropy_loss + ce_loss + global_loss
        # ablation
        # total_loss = consistency_loss + soft_loss - self.entropy_weight*entropy_loss #+ global_loss #+ ce_loss
        
        return total_loss, consistency_loss, entropy_loss


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]
        # soft_loss(anchor.cuda())
        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss

def ce_loss(outputs, targets, sel=None):
    targets = targets.detach()
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * targets
    loss_vec = - (final_outputs).sum(dim=1)
    if sel is None:
        average_loss = loss_vec.mean()
    else:
        average_loss = loss_vec[sel].mean()
    # take mean on selected examples
    return loss_vec, average_loss

class InstanceLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(
        self,
        tau=0.5,
        multiplier=2,
        distributed=False,
        alpha=0.99,
        gamma=0.5,
        cluster_num=10,
    ):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.alpha = alpha
        self.gamma = gamma
        self.cluster_num = cluster_num

    @torch.no_grad()
    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        batch_size = c.shape[0]
        device = c.device
        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
        tmp = torch.arange(0, batch_size).to(device)

        prediction = c.argmax(dim=1)
        confidence = c.max(dim=1).values
        unconfident_pred_index = confidence < self.alpha
        pseudo_per_class = np.ceil(batch_size / self.cluster_num * self.gamma).astype(int)

        for i in range(self.cluster_num):
            class_idx = prediction == i
            if class_idx.sum() == 0:
                continue
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_label_nxt[idx] = i

        todo_index = pseudo_label_cur == -1
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1
        return pseudo_label_nxt.cpu(), index

    def forward(self, z_i, z_j, pseudo_label):
        n = z_i.shape[0]
        assert n % self.multiplier == 0

        invalid_index = pseudo_label == -1
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(
            z_i.device
        )
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().to(z_i.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()

        contrast_count = self.multiplier
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.tau
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask_with_eye = mask | mask_eye.bool()
        # mask = torch.cat(mask)
        mask = mask.repeat(anchor_count, contrast_count)
        mask_eye = mask_eye.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n * anchor_count).view(-1, 1).to(z_i.device),
            0,
        )
        logits_mask *= 1 - mask
        mask_eye = mask_eye * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_eye * log_prob).sum(1) / mask_eye.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, n).mean()

        return instance_loss

class ClusterBoostingLoss(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ClusterBoostingLoss, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, epoch, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold
        # import pdb; pdb.set_trace()
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)
        
        start_ratio = 0.7
        ratio = start_ratio + start_ratio*(1-(200-epoch)/200)
        # import pdb; pdb.set_trace()
        pseudo_per_class = np.ceil(b/c * ratio).astype(int)
        pseudo_labels = -torch.ones(b, dtype=torch.long).cuda()
        tmp = torch.arange(0, b).cuda()
        
        for i in range(c):
            class_idx = target == i
            if class_idx.sum() == 0:
                continue
            confidence_class = max_prob[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_labels[idx] = i
        
        mask = pseudo_labels != -1
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)
        # import pdb; pdb.set_trace()
        # Inputs are strongly augmented anchors
        # input_ = anchors_strong
        input_ = self.softmax(anchors_strong) 

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            # import pdb; pdb.set_trace()
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        return loss
    
class ClusterLossBoostV2(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, cluster_num=10):
        super().__init__()
        self.cluster_num = cluster_num
        self.apply_class_balancing = True

    def forward(self, c, pseudo_label):
        pseudo_index = pseudo_label != -1
        pesudo_label_all = pseudo_label[pseudo_index]
        idx, counts = torch.unique(pesudo_label_all, return_counts=True)
        freq = pesudo_label_all.shape[0] / counts.float()
        weight = torch.ones(self.cluster_num).to(c.device)
        weight[idx] = freq
        pseudo_index = pseudo_label != -1
        # import pdb; pdb.set_trace()
        
        input_ = c[pseudo_index]
        # import pdb; pdb.set_trace()
        # Loss
        return F.cross_entropy(input_, pesudo_label_all, weight = weight, reduction = 'mean')
    
    
class SoftInstanceLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(
        self,
        tau=0.5,
        alpha=0.99,
        gamma=0.5,
        cluster_num=10,
    ):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.cluster_num = cluster_num

    @torch.no_grad()
    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        batch_size = c.shape[0]
        device = c.device
        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
        tmp = torch.arange(0, batch_size).to(device)

        prediction = c.argmax(dim=1)
        confidence = c.max(dim=1).values
        unconfident_pred_index = confidence < self.alpha
        pseudo_per_class = np.ceil(batch_size / self.cluster_num * self.gamma).astype(int)

        for i in range(self.cluster_num):
            class_idx = prediction == i
            if class_idx.sum() == 0:
                continue
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_label_nxt[idx] = i

        todo_index = pseudo_label_cur == -1
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1
        return pseudo_label_nxt.cpu(), index

    def forward(self, z_i, z_j, z_i_prob, z_j_prob, pseudo_label):
        n = z_i.shape[0]

        invalid_index = pseudo_label == -1
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(
            z_i.device
        )
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().to(z_i.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()

        concat_features = torch.cat((z_i, z_j), dim=0)

        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases


        concat_probs = torch.cat((z_i_prob, z_j_prob), dim=0)
        # import pdb; pdb.set_trace()
        instance_loss = soft_neighbors_loss_L2(concat_features, concat_features, concat_probs, concat_probs, boosting=True)
        # import pdb; pdb.set_trace()
        instance_loss = instance_loss * mask
        
        return instance_loss.mean()