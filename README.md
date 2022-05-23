# Fast Visual BERT (简称: FVBERT)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE)
![Python 3.6, 3.7](https://img.shields.io/badge/python-3.7-green.svg)

调研Transformer技术，综合PyTorch, bertviz, Bert-Chinese-Text-Classification-Pytorch, BERT-pytorch, fast-bert等开源项目，研究BERT形成的Fast Visual BERT项目，仅供学习研究。欢迎参考！

## 可视化效果
点开 ipynb 文件查看

## 中文文本分类
+ 训练BERT模型
>python zh_cls_run.py --model zh_cls_bert
+ 训练ERNIE模型
>python zh_cls_run.py --model zh_cls_ernie

## fast_bert
+ 需要NVIDIA Apex
+ 点看[详情](README_fast_bert.md)

## 致谢
+ https://github.com/pytorch/pytorch
+ https://github.com/codertimo/BERT-pytorch
+ https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch
+ https://github.com/jessevig/bertviz
+ https://github.com/utterworks/fast-bert