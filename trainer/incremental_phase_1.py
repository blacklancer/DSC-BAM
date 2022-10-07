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
from utils.loss import *


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


def SCkd(a, b, device=None, normalize=True):
    assert a.shape == b.shape, (a.shape, b.shape)
    a = torch.pow(a, 2)
    b = torch.pow(b, 2)

    a_c = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
    b_c = b.sum(dim=1).view(b.shape[0], -1)
    if normalize:
        a_c = F.normalize(a_c, dim=1, p=2)
        b_c = F.normalize(b_c, dim=1, p=2)
    a_c = a_c.to(device)
    b_c = b_c.to(device)
    # loss_c = nn.SmoothL1Loss(reduction='mean')(a_c, b_c.detach())
    loss_c = nn.CosineEmbeddingLoss()(a_c, b_c.detach(), torch.ones(a_c.shape[0]).to(device))

    a_s = a.sum(dim=(2, 3)).view(a.shape[0], -1)  # shape of (b, c)
    b_s = b.sum(dim=(2, 3)).view(b.shape[0], -1)
    if normalize:
        a_s = F.normalize(a_s, dim=1, p=2)
        b_s = F.normalize(b_s, dim=1, p=2)
    a_s = a_s.to(device)
    b_s = b_s.to(device)
    # loss_s = nn.SmoothL1Loss(reduction='mean')(a_s, b_s.detach())
    loss_s = nn.CosineEmbeddingLoss()(a_s, b_s.detach(), torch.ones(a_s.shape[0]).to(device))

    loss = (loss_c + loss_s)
    return loss


def pod(list_attentions_a, list_attentions_b ,device = None, normalize=True, memory_flags=None, only_old=False):
    assert len(list_attentions_a) == len(list_attentions_b)
    a = list_attentions_a
    b = list_attentions_b
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
    a = torch.cat([a_h, a_w], dim=-1)
    b = torch.cat([b_h, b_w], dim=-1)
    if normalize:
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)
    a = a.to(device)
    b = b.to(device)
    layer_loss = nn.CosineEmbeddingLoss()(a, b.detach(), torch.ones(a.shape[0]).to(device))
    return layer_loss


def RkdAngleSC(a, b):
    a = torch.pow(a, 2)
    b = torch.pow(b, 2)
    a_c = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
    b_c = b.sum(dim=1).view(b.shape[0], -1)
    a_c = F.normalize(a_c, dim=1, p=2)
    b_c = F.normalize(b_c, dim=1, p=2)
    loss_c = RkdAngle(a_c, b_c)
    a_s = a.sum(dim=(2, 3)).view(a.shape[0], -1)  # shape of (b, c)
    b_s = b.sum(dim=(2, 3)).view(b.shape[0], -1)
    a_s = F.normalize(a_s, dim=1, p=2)
    b_s = F.normalize(b_s, dim=1, p=2)
    loss_s = RkdAngle(a_s, b_s)
    return loss_c + loss_s


def RkdAngle(student, teacher):
    td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
    norm_td = F.normalize(td, p=2, dim=2)
    t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
    sd = (student.unsqueeze(0) - student.unsqueeze(1))
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


def RkdDistanceSC(a, b):
    a = torch.pow(a, 2)
    b = torch.pow(b, 2)
    a_c = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
    b_c = b.sum(dim=1).view(b.shape[0], -1)
    a_c = F.normalize(a_c, dim=1, p=2)
    b_c = F.normalize(b_c, dim=1, p=2)
    loss_c = RkdDistance(a_c, b_c)
    a_s = a.sum(dim=(2, 3)).view(a.shape[0], -1)  # shape of (b, c)
    b_s = b.sum(dim=(2, 3)).view(b.shape[0], -1)
    a_s = F.normalize(a_s, dim=1, p=2)
    b_s = F.normalize(b_s, dim=1, p=2)
    loss_s = RkdDistance(a_s, b_s)
    return loss_c + loss_s

def RkdDistance(student, teacher):
    t_d = pdist(teacher, squared=False)
    mean_td = t_d[t_d > 0].mean()
    t_d = t_d / mean_td
    d = pdist(student, squared=False)
    mean_d = d[d > 0].mean()
    d = d / mean_d
    loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
    return loss


def incremental_phase_1(args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, is_start_iteration, iteration, lamda,
            fix_bn=False, weight_per_class=None, device=None):

    # print(tg_model)

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

    contra_criterion = SupConLoss()

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
        train_loss3 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())
        print("lamda=", lamda)

        train_loss31 = 0
        train_loss32 = 0
        train_loss33 = 0
        train_loss34 = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            bsz = targets.size(0)
            # inputs[0] -> origin, inputs[1] -> augmented
            inputs = torch.cat([inputs[0], inputs[1]], dim=0)
            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.tensor(targets, dtype=torch.long)

            tg_optimizer.zero_grad()
            outputs, feat_list = tg_model(inputs)
            # 需要截取，后面增强的与预测结果没用
            outputs = outputs[:bsz]

            temp_targets = targets.detach().cpu()
            new_index = np.array([iteration * 10 <= target for target in temp_targets])
            old_index = np.array([target < iteration * 10 for target in temp_targets])

            if is_start_iteration:
                # 对比的损失
                c_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    c_loss += contra_criterion(features, targets) * 1e-1

                loss1 = c_loss
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss3 = 0
            else:
                ref_outputs, ref_feat_list = ref_model(inputs)

                # 对比的损失
                c_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    c_loss += contra_criterion(features, targets) * 1e-1
                loss1 = c_loss

                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                cur_features_3o, ref_features_3o = cur_features_3[:bsz], ref_features_3[:bsz]
                cur_features_3a, ref_features_3a = cur_features_3[bsz:], ref_features_3[bsz:]

                # cur_old_feature = cur_features[old_index]
                # ref_old_feature = ref_features[old_index]
                # loss11 = RkdAngleSC(cur_features_3[old_index], ref_features_3[old_index]) * 1
                # loss12 = RkdDistanceSC(cur_features_3[old_index], ref_features_3[old_index]) * 1
                # loss13 = SCkd(cur_features_3[old_index], ref_features_3[old_index], device) * 1
                # loss1 = (loss11 + loss12 + loss13) * lamda

                loss31 = pod(cur_features_3o[old_index], ref_features_3o[old_index], device)
                loss32 = pod(cur_features_3a[old_index], ref_features_3a[old_index], device)
                loss33 = pod(cur_features_3o[new_index], ref_features_3o[new_index], device)
                loss34 = pod(cur_features_3a[new_index], ref_features_3a[new_index], device)
                loss3 = (loss31*1 + loss32*1 + loss33*0 + loss34*0) * lamda
                print(loss3.item(), loss31.item(), loss32.item(), loss33.item(), loss34.item())

            loss = loss1 + loss2 + loss3
            loss.backward()
            tg_optimizer.step()

            # 记录各个损失值
            train_loss += loss.item()
            if not is_start_iteration:
                train_loss3 += loss3.item()
                train_loss31 = loss31.item()
                train_loss32 = loss32.item()
                train_loss33 = loss33.item()
                train_loss34 = loss34.item()
            else:
                train_loss3 += loss3
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Train set: {}, '
              'Train Loss1: {:.4f},'
              'Train Loss2: {:.4f},'
              'Train Loss3: {:.4f},'
              'Train Loss31: {:.4f},Train Loss32: {:.4f},Train Loss33: {:.4f},Train Loss34: {:.4f},'
              'Train Loss: {:.4f},''Acc: {:.4f}'
              .format(len(trainloader),
                      train_loss1/(batch_idx+1),
                      train_loss2/(batch_idx+1),
                      train_loss3/(batch_idx+1),
                      train_loss31/(batch_idx+1), train_loss32/(batch_idx+1), train_loss33/(batch_idx+1), train_loss34/(batch_idx+1),
                      train_loss/(batch_idx+1), 100.*correct/total))

        #eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = torch.tensor(targets, dtype=torch.long)
                # outputs = tg_model(inputs)
                outputs, feat_list = tg_model(inputs)
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