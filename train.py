#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: blacklancer
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Copyright (c) 2022
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

""" Main function for this project. """
import os
import argparse
import sys
import numpy as np
from trainer.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument('--gpu',
                        default='0', help='the index of GPU')
    parser.add_argument('--dataset',
                        default='cifar100', type=str, choices=['cifar100', 'imagenet_sub', 'imagenet'])
    parser.add_argument('--ckp_prefix',
                        default=os.path.basename(sys.argv[0])[:-3], type=str, help='Checkpoint prefix')
    parser.add_argument('--epochs',
                        # default = 160,这里改一下
                        default=80 * 1, type=int, help='Epochs')
    parser.add_argument('--train_batch_size',
                        default=128, type=int, help='the batch size for train loader')
    parser.add_argument('--test_batch_size',
                        default=100, type=int, help='the batch size for test loader')
    parser.add_argument('--eval_batch_size',
                        default=128, type=int, help='the batch size for validation loader')
    parser.add_argument('--num_workers',
                        default=1, type=int, help='the number of workers for loading data')
    parser.add_argument('--resume',
                        action='store_true',
                        default=True, help='resume from checkpoint')
    parser.add_argument('--random_seed',
                        default=2022, type=int, help='random seed')

    ### Incremental learning parameters
    parser.add_argument('--num_classes',
                        default=100, type=int)
    parser.add_argument('--nb_cl_fg',
                        default=50, type=int, help='the number of classes in first group')
    parser.add_argument('--nb_cl',
                        default=10, type=int, help='Classes per group')
    parser.add_argument('--nb_protos',
                        default=20, type=int, help='Number of prototypes per class at the end')
    parser.add_argument('--nb_runs',
                        default=1, type=int, help='Number of runs (random ordering of classes at each run)')
    parser.add_argument('--fix_budget',
                        action='store_true', default=True, help='fix budget')

    ### General learning parameters
    parser.add_argument('--base_lr',
                        default=0.1, type=float, help='base learning rate')
    parser.add_argument('--lr_factor',
                        default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--custom_weight_decay',
                        default=5e-4, type=float,help='weight decay parameter for the optimizer')
    parser.add_argument('--custom_momentum',
                        default=0.9, type=float, help='momentum parameter for the optimizer')

    ### Special learning parameters
    parser.add_argument('--T',
                        default=2, type=float, help='Temperature for distillation')
    parser.add_argument('--beta',
                        default=0.25, type=float, help='Beta for distillation')
    parser.add_argument('--the_lambda',
                        default=5, type=float, help='Lamda for LF')
    parser.add_argument('--adapt_lamda',
                        action='store_true', default=True, help='Adaptively change lamda')

    parser.add_argument('--fix_old',
                        action='store_true', default=True, help='Fix old class weight')
    parser.add_argument('--feature_distillation',
                        action='store_true', default=True, help='Feature distillation')
    parser.add_argument('--try_new',
                        action='store_true', default=True, help='Try something new')

    the_args = parser.parse_args()

    # the_args.ckp_prefix = '{}_{}_nfg_{}_ncl_{}_nproto_{}'.format(
    #          the_args.ckp_prefix, the_args.dataset, the_args.nb_cl_fg, the_args.nb_cl, the_args.nb_protos)
    # Fix the random seed
    np.random.seed(the_args.random_seed)

    # Checke the number of classes, ensure they are reasonable
    assert (the_args.nb_cl_fg % the_args.nb_cl == 0)
    assert (the_args.nb_cl_fg >= the_args.nb_cl)

    # Print the parameters
    print(the_args)

    # Set GPU index
    os.environ['CUDA_VISIBLE_DEVICES'] = the_args.gpu
    print('Using gpu:', the_args.gpu)

    # Set the trainer and start training
    trainer = Trainer(the_args)
    trainer.train()
