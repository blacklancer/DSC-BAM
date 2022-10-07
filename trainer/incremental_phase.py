# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: blacklancer
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Copyright (c) 2022
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 记录传递过程中的feature
cur_features = []
ref_features = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

# 记录传递过程中layer1的feature
cur_features_1 = []
ref_features_1 = []
def get_ref_features_1(self, inputs, outputs):
    global ref_features_1
    ref_features_1 = outputs

def get_cur_features_1(self, inputs, outputs):
    global cur_features_1
    cur_features_1 = outputs

# 记录传递过程中layer2的feature
cur_features_2 = []
ref_features_2 = []
def get_ref_features_2(self, inputs, outputs):
    global ref_features_2
    ref_features_2 = outputs

def get_cur_features_2(self, inputs, outputs):
    global cur_features_2
    cur_features_2 = outputs

# 记录传递过程中layer3的feature
cur_features_3 = []
ref_features_3 = []
def get_ref_features_3(self, inputs, outputs):
    global ref_features_3
    ref_features_3 = outputs

def get_cur_features_3(self, inputs, outputs):
    global cur_features_3
    cur_features_3 = outputs

def pod(list_attentions_a, list_attentions_b ,device = None, normalize=True, memory_flags=None, only_old=False):
    # Pooled Output Distillation.
    # :param list_attentions_a: A list of attention maps, each of shape (b, c, w, h).
    # :param list_attentions_b: A list of attention maps, each of shape (b, c, w, h).
    # :param memory_flags: Integer flags denoting exemplars.
    # :param only_old: Only apply loss to exemplars.
    # :return: A float scalar loss.
    assert len(list_attentions_a) == len(list_attentions_b)

    a = list_attentions_a
    b = list_attentions_b
    # shape of (b, n, w, h)
    assert a.shape == b.shape, (a.shape, b.shape)

    if only_old:
        a = a[memory_flags]
        b = b[memory_flags]

    a = torch.pow(a, 2)
    b = torch.pow(b, 2)

    a_h = a.sum(dim=3).view(a.shape[0], -1)# shape of (b, c * w)
    b_h = b.sum(dim=3).view(b.shape[0], -1)
    a_w = a.sum(dim=2).view(a.shape[0], -1)# shape of (b, c * h)
    b_w = b.sum(dim=2).view(b.shape[0], -1)
    # a_c = a.sum(dim=1).view(a.shape[0], -1)# shape of (b, w * h)
    # b_c = b.sum(dim=1).view(b.shape[0], -1)
    # a_h = a.sum(dim=(1, 2)).view(a.shape[0], -1)  # shape of (b, h)
    # b_h = b.sum(dim=(1, 2)).view(b.shape[0], -1)
    # a_w = a.sum(dim=(1, 3)).view(a.shape[0], -1)  # shape of (b, w)
    # b_w = b.sum(dim=(1, 3)).view(b.shape[0], -1)
    # a_c = a.sum(dim=(2, 3)).view(a.shape[0], -1)  # shape of (b, c)
    # b_c = b.sum(dim=(2, 3)).view(b.shape[0], -1)
    a = torch.cat([a_h, a_w], dim=-1)
    b = torch.cat([b_h, b_w], dim=-1)
    # a = torch.cat([a, a_c], dim=-1)
    # b = torch.cat([b, b_c], dim=-1)

    if normalize:
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)
    a = a.to(device)
    b = b.to(device)
    layer_loss = nn.CosineEmbeddingLoss()(a, b.detach(), torch.ones(a.shape[0]).to(device))
    # layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))

    return layer_loss

def SCkd(a, b, device=None, normalize=True):
    assert a.shape == b.shape, (a.shape, b.shape)
    a = torch.pow(a, 2)
    b = torch.pow(b, 2)

    # a_s = a.sum(dim=(2, 3))
    # b_s = b.sum(dim=(2, 3))
    # if normalize:
    #     a_s = F.normalize(a_s, dim=1, p=2)
    #     b_s = F.normalize(b_s, dim=1, p=2)
    # a_s = torch.abs(a_s)
    # b_s = torch.abs(b_s)
    # zero = torch.zeros_like(a_s)
    # # 应该是以ref_model中的重要性作为标准
    # a_s = torch.where(b_s > 0.1, a_s, zero)
    # b_s = torch.where(b_s > 0.1, b_s, zero)
    # a_s = a_s.unsqueeze(2).unsqueeze(3)
    # b_s = b_s.unsqueeze(2).unsqueeze(3)
    # a_s = a_s.repeat(1, 1, a.shape[2], a.shape[3])
    # b_s = b_s.repeat(1, 1, b.shape[2], b.shape[3])
    # a_s = a_s.to(device)
    # b_s = b_s.to(device)
    # a = torch.mul(a, a_s)
    # b = torch.mul(b, b_s)

    a_s = torch.abs(a)
    b_s = torch.abs(b)
    zero = torch.zeros_like(a_s)
    # 应该是以ref_model中的重要性作为标准
    a = torch.where(b_s > 0.1, a, zero)
    b = torch.where(b_s > 0.1, b, zero)

    a_h = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
    b_h = b.sum(dim=3).view(b.shape[0], -1)
    a_w = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
    b_w = b.sum(dim=2).view(b.shape[0], -1)
    a = torch.cat([a_h, a_w], dim=-1)
    b = torch.cat([b_h, b_w], dim=-1)
    if normalize:
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)
    a = a.to(device)
    b = b.to(device)
    loss = nn.CosineEmbeddingLoss()(a, b.detach(), torch.ones(a.shape[0]).to(device))
    # loss = F.smooth_l1_loss(a, b, reduction='elementwise_mean')*a.shape[0]

    # a_t = a.sum(dim=(2, 3)).view(a.shape[0], -1)  # shape of (b, c)
    # b_t = b.sum(dim=(2, 3)).view(b.shape[0], -1)
    # a_t = F.normalize(a_t, dim=1, p=2)
    # b_t = F.normalize(b_t, dim=1, p=2)
    # a_t = a_t.to(device)
    # b_t = b_t.to(device)
    # loss = nn.CosineEmbeddingLoss()(a_t, b_t.detach(), torch.ones(a.shape[0]).to(device))
    # loss = F.smooth_l1_loss(a, b, reduction='elementwise_mean')

    return loss


def scd(list_attentions_a, list_attentions_b ,device = None, normalize=True, memory_flags=None, only_old=False):
    assert len(list_attentions_a) == len(list_attentions_b)

    a = list_attentions_a
    b = list_attentions_b
    assert a.shape == b.shape, (a.shape, b.shape)

    if only_old:
        a = a[memory_flags]
        b = b[memory_flags]

    a = torch.pow(a, 2)
    b = torch.pow(b, 2)
    a_c = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
    b_c = b.sum(dim=1).view(b.shape[0], -1)
    a_s = a.sum(dim=(2, 3)).view(a.shape[0], -1)  # shape of (b, c)
    b_s = b.sum(dim=(2, 3)).view(b.shape[0], -1)
    a = torch.cat([a_c, a_s], dim=-1)
    b = torch.cat([b_c, b_s], dim=-1)
    if normalize:
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)
    a = a.to(device)
    b = b.to(device)
    # layer_loss = nn.CosineEmbeddingLoss()(a, b.detach(), torch.ones(a.shape[0]).to(device))
    # layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
    # layer_loss = nn.MSELoss()(a, b.detach()) * 1
    layer_loss = F.smooth_l1_loss(a, b, reduction='elementwise_mean')

    return layer_loss


def RkdAngle(student, teacher, device = None):
    # N x C
    # N x N x C
    td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
    norm_td = F.normalize(td, p=2, dim=2)
    t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = (student.unsqueeze(0) - student.unsqueeze(1))
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
    return loss


def RkdAngleSD(student, teacher, device = None):
    a = student
    b = teacher
    a = torch.pow(a, 2)
    b = torch.pow(b, 2)
    a_s = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
    b_s = b.sum(dim=1).view(b.shape[0], -1)
    a_c = a.sum(dim=(2, 3)).view(a.shape[0], -1)  # shape of (b, c)
    b_c = b.sum(dim=(2, 3)).view(b.shape[0], -1)
    a = torch.cat([a_s, a_c], dim=-1)
    b = torch.cat([b_s, b_c], dim=-1)
    a = F.normalize(a, dim=1, p=2)
    b = F.normalize(b, dim=1, p=2)

    td = (b.unsqueeze(0) - b.unsqueeze(1))
    norm_td = F.normalize(td, p=2, dim=2)
    t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = (a.unsqueeze(0) - a.unsqueeze(1))
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
    return loss


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def RkdDistance(student, teacher, device = None):
    t_d = pdist(teacher, squared=False)
    mean_td = t_d[t_d > 0].mean()
    t_d = t_d / mean_td

    d = pdist(student, squared=False)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
    return loss

def RkdDistanceSD(student, teacher, device = None):
    a = student
    b = teacher
    a = torch.pow(a, 2)
    b = torch.pow(b, 2)
    a_s = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
    b_s = b.sum(dim=1).view(b.shape[0], -1)
    a_c = a.sum(dim=(2, 3)).view(a.shape[0], -1)  # shape of (b, c)
    b_c = b.sum(dim=(2, 3)).view(b.shape[0], -1)
    a = torch.cat([a_s, a_c], dim=-1)
    b = torch.cat([b_s, b_c], dim=-1)
    a = F.normalize(a, dim=1, p=2)
    b = F.normalize(b, dim=1, p=2)

    t_d = pdist(b, squared=False)
    mean_td = t_d[t_d > 0].mean()
    t_d = t_d / mean_td

    d = pdist(a, squared=False)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
    return loss



def incremental_phase(args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, is_start_iteration, iteration, lamda,
            fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = tg_model.fc.out_features
    handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
    handle_cur_features_1 = tg_model.layer1.register_forward_hook(get_cur_features_1)
    handle_cur_features_2 = tg_model.layer2.register_forward_hook(get_cur_features_2)
    handle_cur_features_3 = tg_model.layer3.register_forward_hook(get_cur_features_3)

    if not is_start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_ref_features_1 = ref_model.layer1.register_forward_hook(get_ref_features_1)
        handle_ref_features_2 = ref_model.layer2.register_forward_hook(get_ref_features_2)
        handle_ref_features_3 = ref_model.layer3.register_forward_hook(get_ref_features_3)

    epochs = args.epochs
    T = args.T
    beta = args.beta
    for epoch in range(epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())


        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.tensor(targets, dtype=torch.long)
            # print(targets)

            temp_targets = targets.detach().cpu()
            new_index = np.array([iteration*10 <= target for target in temp_targets])
            old_index = np.array([target < iteration*10 for target in temp_targets])

            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)

            if is_start_iteration:
                loss1 = 0
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_outputs = ref_model(inputs)

                cur_old_feature = cur_features[old_index]
                ref_old_feature = ref_features[old_index]

                loss11 = RkdAngle(cur_old_feature, ref_old_feature, device) * 2
                loss12 = RkdDistance(cur_old_feature, ref_old_feature, device) * 1
                loss13 = pod(cur_features_3[old_index], ref_features_3[old_index], device) * 1
                loss1 = (loss11 + loss12 + loss13) * lamda

                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

            loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

            # 记录各个损失值
            train_loss += loss.item()
            if not is_start_iteration:
                train_loss1 += loss1.item()
                # train_loss11 += loss11.item()
                # train_loss12 += loss12.item()
                # train_loss13 += loss13.item()
            else:
                train_loss1 += loss1
            train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Train set: {}, Train Loss1: {:.4f},'
              # 'Train Loss11: {:.4f},Train Loss12: {:.4f},Train Loss13: {:.4f},'
              'Train Loss2: {:.4f},Train Loss: {:.4f},''Acc: {:.4f}'
              .format(len(trainloader),train_loss1/(batch_idx+1),
                      # train_loss11/(batch_idx+1),train_loss12/(batch_idx+1),train_loss13/(batch_idx+1),
                      train_loss2/(batch_idx+1),train_loss/(batch_idx+1), 100.*correct/total))

        #eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = torch.tensor(targets, dtype=torch.long)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    # print("Removing register_forward_hook")
    handle_cur_features.remove()
    handle_cur_features_1.remove()
    handle_cur_features_2.remove()
    handle_cur_features_3.remove()
    if not is_start_iteration:
        handle_ref_features.remove()
        handle_ref_features_1.remove()
        handle_ref_features_2.remove()
        handle_ref_features_3.remove()

    return tg_model