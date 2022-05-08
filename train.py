"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

train.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from options import Options
from lib.data import load_data
from lib.timegan import TimeGAN


def train(opt):
    """ Training
    """


    # LOAD DATA
    ori_data = load_data(opt)

    # LOAD MODEL
    model = TimeGAN(opt, ori_data)

    # TRAIN MODEL
    model.train()


if __name__ == '__main__':
    import argparse

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['sine', 'stock', 'energy', 'czb'],
        default='czb',
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=24,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru', 'lstm', 'lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=16,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=10000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=8,
        type=int)
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=1,
        type=int)
    parser.add_argument(
        '--w_gamma',
        help='weight of the generator unsupervised loss for embedding',
        default=1,
        type=int)
    parser.add_argument(
        '--lr',
        help='learning rate',
        default=1,
        type=int)
    parser.add_argument(
        '--isTrain',
        help='if choosing training mode',
        default=True,
        type=int)
    parser.add_argument(
        '--resume',
        help='if resume training',
        default=False,
        type=int)
    parser.add_argument(
        '--beta1',
        help='coefficients used for computing running averages of gradient',
        default=0.9,
        type=int)
    parser.add_argument(
        '--w_g',
        help='weight of the moment loss among generator loss',
        default=100,
        type=int)
    parser.add_argument(
        '--z_dim',
        help='The number of expected features in the input',
        default=17,
        type=int)
    parser.add_argument(
        '--device',
        help='device used for training',
        default='cpu',
        type=int)
    parser.add_argument(
        '--manualseed',
        help='random state',
        default='42',
        type=int)
    parser.add_argument(
        '--outf',
        help='output file path',
        default='data',
        type=int)
    parser.add_argument(
        '--name',
        help='output file path',
        default='czb',
        type=int)




    args = parser.parse_args()
    train(args)
