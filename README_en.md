# Wireless Channel AI
[![Travis](https://img.shields.io/travis/Lmy0217/WirelessChannelAI.svg?branch=master&label=Travis+CI)](https://www.travis-ci.org/Lmy0217/WirelessChannelAI) [![CircleCI](https://img.shields.io/circleci/project/github/Lmy0217/WirelessChannelAI.svg?branch=master&label=CircleCI)](https://circleci.com/gh/Lmy0217/WirelessChannelAI) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Lmy0217/WirelessChannelAI/pulls) [![简体中文](https://img.shields.io/badge/README-简体中文-blue.svg)](README.md)

* This repo is code of [2019 年中国研究生数学建模竞赛 A 题](https://developer.huaweicloud.com/competition/competitions/1000013923/introduction). **The code will be open source after the competition**.

## Dependency
- [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) with GPU
- Python 3 with packages [tensorFlow-gpu==2.0.0rc1](https://github.com/tensorflow/tensorflow) installed
- Matlab

## Prerequisite
- Download this repo
  ```bash
  git clone https://github.com/Lmy0217/WirelessChannelAI.git
  cd WirelessChannelAI
  ```

- Install requirements
  ```bash
  pip3 install -r requirements.txt
  ```

## Dataset

- Download [dataset](https://developer.huaweicloud.com/competition/competitions/1000013923/circumstances) and extract the zip file in the folder `data` (now, this folder should contain two folder named 'train_set' and 'test_set' respectively)

- Run combine and prepocess code in the `data` folder
  ```bash
  cd data
  python3 comp.py
  matlab -nodesktop -nosplash -r "poccess;exit;"
  ```

## Model
- Training and testing model
  ```bash
  python3 model.py
  ```

- Trained model will be saved in the `model` folder, **will cover the pre-training model**

## Pre-trained Model
- Saved in the `model` folder

## Performance
- Three quarters of the data set is used as train set and the remaining quarter as test set.

|Model|Offline RMSE|Online RMSE|
|-|-|-|
|FC-5|≈ 9.74|9.7588|
|FC-6|≈ 9.53|9.4514|
|ResNet-8|≈ 9.43|-|

## Online Verification 

- Upload the `model` folder as the model directory

- Upload the `test_set` folder under the `data` folder as the test set directory

- Create model and deploy batch service in [ModelArts](https://console.huaweicloud.com/modelarts)

- Submit the model to the competition

## License
The code is licensed with the [MIT](LICENSE) license.