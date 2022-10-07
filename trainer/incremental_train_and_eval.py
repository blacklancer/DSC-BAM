#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: blacklancer
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Copyright (c) 2022
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
import torch.nn.functional as F


def incremental_train_and_eval(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, is_start_iteration, T, beta, \
            fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not is_start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features

    for epoch in range(epochs):
        #train
        tg_model.train()
        # Fix the batch norm parameters according to the config
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    
        # 初始化所有的loss值
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0

        tg_lr_scheduler.step()
        # print('\nEpoch: %d, LR: ' % epoch, end='')
        # print(tg_lr_scheduler.get_lr())

        # 训练过程
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.tensor(targets, dtype=torch.long)

            # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)

            if is_start_iteration:
                # 0-th 阶段，是非增量阶段，不需要蒸馏，只要计算分类损失就好了
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())
            else:
                # i-th 阶段，增量阶段，需要考虑蒸馏损失
                ref_outputs = ref_model(inputs)
                # 蒸馏损失
                loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                    F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes
                # 分类损失
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())
                loss = loss1 + loss2

            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if not is_start_iteration:
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # if is_start_iteration:
        #     print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
        #         len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        # else:
        #     print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f},\
        #         Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
        #         train_loss1/(batch_idx+1), train_loss2/(batch_idx+1),
        #         train_loss/(batch_idx+1), 100.*correct/total))

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

                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
        #     len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
        
    return tg_model