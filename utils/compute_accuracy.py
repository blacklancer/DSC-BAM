##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" The functions that compute the accuracies """
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from utils.misc import *


def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y


def compute_accuracy(tg_model, tg_feature_model, class_means, evalloader, scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tg_model.eval()
    tg_feature_model.eval()

    correct = 0
    correct_nme = 0
    correct_nms = 0
    total = 0

    # correct是使用tg_model的结果，全连接分类器，其实改成correct_fc更好
    # correct_nme是使用tg_feature_model的结果，NCM，不过使用的是mean-of-exemplar
    # correct_nms是使用tg_feature_model的结果，NCM分类器，使用的是mean-of-all samples，作为对比实验

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):

            inputs, targets = inputs.to(device), targets.to(device)

            # total统计现在为止的数据量，一个总数
            total += targets.size(0)

            # compute score for FC
            # outputs = tg_model(inputs)
            outputs, _ = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)

            # 这个不知道干啥的
            if scale is not None:
                assert (scale.shape[0] == 1)
                assert (outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            # 下面两个计算使用的特征都是一样的，重复使用就好
            outputs_feature = (np.squeeze(tg_feature_model(inputs))).cpu().numpy()

            # Compute score for NME
            sqd_nme = cdist(class_means[:, :, 0].T, outputs_feature, 'sqeuclidean')
            score_nme = torch.from_numpy((-sqd_nme).T).to(device)
            # 这里max就是所有数值中进行比较，缩放不缩放没有大不了的
            _, predicted_nme = score_nme.max(1)
            correct_nme += predicted_nme.eq(targets).sum().item()

            # Compute score for NCM
            sqd_nms = cdist(class_means[:, :, 1].T, outputs_feature, 'sqeuclidean')
            score_nms = torch.from_numpy((-sqd_nms).T).to(device)
            _, predicted_nms = score_nms.max(1)
            correct_nms += predicted_nms.eq(targets).sum().item()
            # print(sqd_icarl.shape, score_icarl.shape, predicted_icarl.shape, \
            # sqd_ncm.shape, score_ncm.shape, predicted_ncm.shape)
    if print_info:
        print("  top 1 accuracy FC           :\t\t{:.2f} %".format(100. * correct / total))
        print("  top 1 accuracy NME          :\t\t{:.2f} %".format(100. * correct_nme / total))
        print("  top 1 accuracy NMS          :\t\t{:.2f} %".format(100. * correct_nms / total))

    fc_acc  = 100. * correct / total
    nme_acc = 100. * correct_nme / total
    nms_acc = 100. * correct_nms / total

    return [fc_acc, nme_acc, nms_acc]
