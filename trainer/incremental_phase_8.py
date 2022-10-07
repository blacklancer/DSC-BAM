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


def ASCkd(cur, ref, tg_model_b, ref_model_b, device=None):
    cur_channel_tensor = tg_model_b.channel_att(cur)
    cur_spatial_tensor = tg_model_b.spatial_att(cur)
    cur_att = F.sigmoid(cur_channel_tensor * cur_spatial_tensor)
    cur_f = (1 + cur_att) * cur

    ref_channel_tensor = ref_model_b.channel_att(ref)
    ref_spatial_tensor = ref_model_b.spatial_att(ref)
    ref_att = F.sigmoid(ref_channel_tensor * ref_spatial_tensor)
    ref_f = (1 + ref_att) * ref

    loss1 = SCkd(cur_att, ref_att, device)
    loss2 = SCkd(cur_f, ref_f, device)
    loss3 = Skd(cur_f, ref_f, device)
    return loss1, loss2, loss3


def Skd(a, b, device=None, normalize=True):
    T = 0.3
    a = torch.pow(a, 2)
    b = torch.pow(b, 2)
    a_c = a.mean(dim=2).view(a.shape[0], -1)  # shape of (b, w * h)
    b_c = b.mean(dim=2).view(b.shape[0], -1)
    if normalize:
        a_c = F.normalize(a_c, dim=1, p=2)
        b_c = F.normalize(b_c, dim=1, p=2)
    a_c = a_c.to(device)
    b_c = b_c.to(device)
    similarity_ac = F.cosine_similarity(a_c.unsqueeze(1), a_c.unsqueeze(0), dim=2)
    similarity_ac = torch.ones_like(similarity_ac) - similarity_ac
    similarity_bc = F.cosine_similarity(b_c.unsqueeze(1), b_c.unsqueeze(0), dim=2)
    similarity_bc = torch.ones_like(similarity_bc) - similarity_bc
    loss_c = nn.KLDivLoss()(F.log_softmax(similarity_ac / T, dim=1), F.softmax(similarity_bc.detach() / T, dim=1)) * \
             a_c.shape[0]
    # a_s = a.mean(dim=3).view(a.shape[0], -1)  # shape of (b, c)
    # b_s = b.mean(dim=3).view(b.shape[0], -1)
    # if normalize:
    #     a_s = F.normalize(a_s, dim=1, p=2)
    #     b_s = F.normalize(b_s, dim=1, p=2)
    # a_s = a_s.to(device)
    # b_s = b_s.to(device)
    # similarity_as = F.cosine_similarity(a_s.unsqueeze(1), a_s.unsqueeze(0), dim=2)
    # similarity_as = torch.ones_like(similarity_as) - similarity_as
    # similarity_bs = F.cosine_similarity(b_s.unsqueeze(1), b_s.unsqueeze(0), dim=2)
    # similarity_bs = torch.ones_like(similarity_bs) - similarity_bs
    # loss_s = nn.KLDivLoss()(F.log_softmax(similarity_as / T, dim=1), F.softmax(similarity_bs.detach() / T, dim=1)) * \
    #          a_s.shape[0]
    return loss_c


def SCkd(a, b, device=None, normalize=True):
    a = torch.pow(a, 2)
    b = torch.pow(b, 2)
    a_c = a.mean(dim=2).view(a.shape[0], -1)  # shape of (b, w * h)
    b_c = b.mean(dim=2).view(b.shape[0], -1)
    if normalize:
        a_c = F.normalize(a_c, dim=1, p=2)
        b_c = F.normalize(b_c, dim=1, p=2)
    a_c = a_c.to(device)
    b_c = b_c.to(device)
    loss_c = nn.CosineEmbeddingLoss()(a_c, b_c.detach(), torch.ones(a_c.shape[0]).to(device))
    # a_s = a.mean(dim=3).view(a.shape[0], -1)  # shape of (b, c)
    # b_s = b.mean(dim=3).view(b.shape[0], -1)
    # if normalize:
    #     a_s = F.normalize(a_s, dim=1, p=2)
    #     b_s = F.normalize(b_s, dim=1, p=2)
    # a_s = a_s.to(device)
    # b_s = b_s.to(device)
    # loss_s = nn.CosineEmbeddingLoss()(a_s, b_s.detach(), torch.ones(a_s.shape[0]).to(device))
    # loss = (loss_c + loss_s)
    return loss_c


def incremental_phase_8(args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader,
                        is_start_iteration, iteration, lamda, fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = tg_model.fc.out_features
    handle_cur_features_1 = tg_model.layer1.register_forward_hook(get_cur_features_1)
    handle_cur_features_2 = tg_model.layer2.register_forward_hook(get_cur_features_2)
    handle_cur_features_3 = tg_model.layer3.register_forward_hook(get_cur_features_3)
    if not is_start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        handle_ref_features_1 = ref_model.layer1.register_forward_hook(get_ref_features_1)
        handle_ref_features_2 = ref_model.layer2.register_forward_hook(get_ref_features_2)
        handle_ref_features_3 = ref_model.layer3.register_forward_hook(get_ref_features_3)

    contra_criterion = SupConLoss()

    for epoch in range(args.epochs):
        # train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_loss5 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())

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

                c_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    c_loss += contra_criterion(features, targets) * 1e-1 * (1 - index / len(feat_list))
                loss1 = c_loss * 0.5
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                old_index = torch.tensor(old_index)
                old_index = torch.cat((old_index, old_index), dim=0)
                old_index = old_index.detach().numpy()

                # loss31, loss41, loss51 = ASCkd(cur_features_1, ref_features_1, tg_model.bam1, ref_model.bam1, device)
                # loss32, loss42, loss52 = ASCkd(cur_features_2, ref_features_2, tg_model.bam2, ref_model.bam2, device)
                # loss33, loss43, loss53 = ASCkd(cur_features_3, ref_features_3, tg_model.bam3, ref_model.bam3, device)

                loss31, loss41, loss51 = ASCkd(cur_features_1[old_index], ref_features_1[old_index],
                                               tg_model.bam1, ref_model.bam1, device)
                loss32, loss42, loss52 = ASCkd(cur_features_2[old_index], ref_features_2[old_index],
                                               tg_model.bam2, ref_model.bam2, device)
                loss33, loss43, loss53 = ASCkd(cur_features_3[old_index], ref_features_3[old_index],
                                               tg_model.bam3, ref_model.bam3, device)

                loss3 = (loss31 * 3 + loss32 * 2 + loss33 * 1) * lamda * 0.5
                loss4 = (loss41 * 3 + loss42 * 2 + loss43 * 1) * lamda * 1
                loss5 = (loss51 * 3 + loss52 * 2 + loss53 * 1) * lamda * 0.5

            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            tg_optimizer.step()

            # 记录各个损失值
            train_loss += loss.item()
            if not is_start_iteration:
                train_loss3 += loss3.item()
                train_loss4 += loss4.item()
                train_loss5 += loss5.item()
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
              'Train Loss4: {:.4f},'
              'Train Loss5: {:.4f},'
              'Train Loss: {:.4f},''Acc: {:.4f}'
              .format(len(trainloader),
                      train_loss1 / (batch_idx + 1),
                      train_loss2 / (batch_idx + 1),
                      train_loss3 / (batch_idx + 1),
                      train_loss4 / (batch_idx + 1),
                      train_loss5 / (batch_idx + 1),
                      train_loss / (batch_idx + 1), 100. * correct / total))

        # eval
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
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format( \
            len(testloader), test_loss / (batch_idx + 1), 100. * correct / total))

    # print("Removing register_forward_hook")
    handle_cur_features_1.remove()
    handle_cur_features_2.remove()
    handle_cur_features_3.remove()
    if not is_start_iteration:
        handle_ref_features_1.remove()
        handle_ref_features_2.remove()
        handle_ref_features_3.remove()

    return tg_model