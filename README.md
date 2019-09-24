# Wireless Channel AI
[![Travis](https://img.shields.io/travis/Lmy0217/WirelessChannelAI.svg?branch=master&label=Travis+CI)](https://www.travis-ci.org/Lmy0217/WirelessChannelAI) [![CircleCI](https://img.shields.io/circleci/project/github/Lmy0217/WirelessChannelAI.svg?branch=master&label=CircleCI)](https://circleci.com/gh/Lmy0217/WirelessChannelAI) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Lmy0217/WirelessChannelAI/pulls) [![English](https://img.shields.io/badge/README-English-blue.svg)](README_en.md)

* 本仓库存放 [2019 年中国研究生数学建模竞赛 A 题](https://developer.huaweicloud.com/competition/competitions/1000013923/introduction)代码，**在竞赛结束后开源**。

## 依赖
- 安装 [CUDA](https://developer.nvidia.com/cuda-toolkit) 和 [cuDNN](https://developer.nvidia.com/cudnn) 的 GPU
- Python 3 安装 [tensorFlow-gpu==2.0.0rc1](https://github.com/tensorflow/tensorflow)
- Matlab

## 准备
- 下载本仓库
  ```bash
  git clone https://github.com/Lmy0217/WirelessChannelAI.git
  cd WirelessChannelAI
  ```

- 安装依赖
  ```bash
  pip3 install -r requirements.txt
  ```

## 数据集

- 下载[数据集](https://developer.huaweicloud.com/competition/competitions/1000013923/circumstances)，并将其解压到 `data` 文件夹（现在，这个文件夹应该包含 ‘train_set’ 和 ‘test_set’ 文件夹）

- 在 `data` 文件夹下运行合并、预处理代码
  ```bash
  cd data
  python3 comp.py
  matlab -nodesktop -nosplash -r "poccess;exit;"
  ```

## 模型
- 训练和测试模型
  ```bash
  python3 model.py
  ```

- 训练后模型保存在 `model` 文件夹下，**会覆盖预训练模型**

## 预训练模型
- 保存在 `model` 文件夹

## 性能
- 数据集四分之三作为训练集，其余四分之一作为测试集

|模型|线下 RMSE|线上 RMSE|
|-|-|-|
|FC-5|≈ 9.74|9.7588|
|FC-6|≈ 9.53|9.4514|
|ResNet-8|≈ 9.43|-|

## 线上验证

- 上传 `model` 文件夹作为模型目录

- 上传 `data` 文件夹下的 `test_set` 文件夹作为测试集目录

- 在 [ModelArts](https://console.huaweicloud.com/modelarts) 创建模型，部署批量服务

- 提交模型到比赛

## 许可证
代码在 [MIT](LICENSE) 许可证下开源。