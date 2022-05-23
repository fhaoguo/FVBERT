# coding: utf-8
# @Author     :fenghaoguo
# @Time       :2022/5/23 09:36
# @FileName   :zh_cls_run.py
# @Description:

import argparse
import time
from importlib import import_module

import numpy as np
import torch

from zh_cls_data_util import build_dataset, build_iterator, get_time_dif
from zh_cls_train_eval import train

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a m2: zh_cls_bert, zh_cls_ernie')
args = parser.parse_args()

# dataset = '/content/drive/My Drive/THUCNews'  # 数据集
# pt_path = '/content/drive/My Drive/pt'  # ckpt路径
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    device = 'cuda'
    dataset = '/content/drive/My Drive/THUCNews'  # 数据集
    pt_path = '/content/drive/My Drive/pt'  # ckpt路径
else:
    device = 'cpu'
    dataset = '../../data/THUCNews'  # 数据集
    pt_path = '../../pt'  # ckpt路径

if __name__ == '__main__':

    model_name = args.model  # bert
    x = import_module('model.' + model_name)
    config = x.Config(dataset, pt_path, device)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
