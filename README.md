# CrowdTracker-Pytorch(单摄像头)

[简体中文](./README.md) | [English](./README_en.md)

## 地址导航

+ [→Paddle版地址](https://github.com/ReverseSacle/FairMOT-Paddle-Tracker_Basic)
+ [→FairMot作者(Github)](https://github.com/ifzhang/FairMOT)

## 效果预览

![MOT20-01](./docs/MOT20-01.gif)

## 界面预览

![Interface](./docs/Interface.png)

## 相关介绍

+ [→制作介绍](./docs/Making_Introduction_cn.md)

## 环境要求

+ python3
+ opencv-python
+ DCNv2
+ 已运行的测试平台 → window10
+ 已经配置好的conda环境(所需要的全部环境的整合)  [→OneDrive](https://1drv.ms/u/s!AlYD8lJlPHCIiSrFcXk8xcSq_zLD?e=e51wjQ?download=1)

## 调试运行

+ ` git clone "https://github.com/ReverseSacle/CrowdTracker-Pytorch_Basic.git"`
+ 解压`CrowdTracker-env`环境到`./Anaconda3/envs/`目录下
+ 使用编译器，例如Pycharm，调用此`CrowdTracker-env`环境，再在此根目录中创建一个`models`文件夹，将下面的模型权重压缩包解压到此文件夹中

## 提供的模型权重文件

+ **下载：** 由原作者提供 [→OneDrive](https://1drv.ms/u/s!AlYD8lJlPHCIh22rxkVDfBph2VCM?e=0Tudce?download=1)  默认需放置根目录的models文件夹下
+ **额外缺少的文件：** [→OneDrive](https://1drv.ms/u/s!AlYD8lJlPHCIh2xS1T_M_RBKkTIf?e=iae70F?download=1)  放置在`C:\Users\User name\.cache\torch\hub\checkpoints`

## 基础套件

+ `PyQt5` 	→  界面窗口、按钮组、阈值选择、GPU选择、文件选择与进度条
+ `Pytorch` →  深度学习追踪系统
+ `OpenCV` →  视频和摄像头追踪，播放与暂停

## 更新日志

2021.11.29  添加新分支ByteTrack-Kernel，以ByteTrack核心替换了当前的追踪核心

2022.12.12  分别将ByteTrack追踪核心与FairMot追踪核心的代码进行了精简化，各将代码拆分成了界面、视频追踪、内置摄像头追踪与外置摄像头追踪。整合了LINK2001错误修复环境。
