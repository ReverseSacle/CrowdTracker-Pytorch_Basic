# FairMOT_Pytorch_Tracker(单摄像头)
[简体中文](https://github.com/ReverseSacle/FairMOT-Pytorch-Tracker_Basic/blob/main/README.md) | [English](https://github.com/ReverseSacle/FairMOT-Pytorch-Tracker_Basic/blob/main/README_en.md)

地址导航
---
+ [→Paddle版地址](https://github.com/ReverseSacle/FairMOT-Paddle-Tracker_Basic)
+ [→原作者Github地址](https://github.com/ifzhang/FairMOT)

效果预览
---
![MOT20-01](https://github.com/ReverseSacle/FairMOT-Pytorch-Tracker_Basic/blob/main/docs/MOT20-01.gif)

界面预览
---
![Interface](https://user-images.githubusercontent.com/73418195/126268446-f38053a6-3b1c-4c3f-98c2-afe07030a8ff.png)


相关介绍
---
+ [→制作介绍](https://github.com/ReverseSacle/FairMOT-Pytorch-Tracker_Basic/blob/main/docs/Making_Introduction_cn.md)
+ [→软件使用指南](https://github.com/ReverseSacle/FairMOT-Pytorch-Tracker_Basic/blob/main/docs/The_fuction_of_program_cn.md)


环境要求
---
+ python3
+ OpenCV
+ DCNV2
+ 需要的第三方库 → 请查看[箭头原作者Github地址](https://github.com/ifzhang/FairMOT)的requirements.txt
+ 运行的测试平台 → window10
+ 已经配置好的conda环境(所需要的全部环境的整合) --> **Fairmot-env下载：**[→OneDrive](https://1drv.ms/u/s!AlYD8lJlPHCIh24gf-IZtNYYN4r0?e=qhCxTc?download=1)

调试运行
---
+ ``` git clone "https://github.com/ReverseSacle/FairMOT-Pytorch-Tracker_Basic.git"```
+ 解压Fairmot_env环境到Anaconda3/envs/目录下
+ 使用pycharm，调用此Fairmot-env环境,再在根目录中创建一个**models**文件夹将下面的模型权重压缩包解压到此文件夹


提供的模型权重文件
---
+ **下载：** 由原作者提供[→OneDrive](https://1drv.ms/u/s!AlYD8lJlPHCIh22rxkVDfBph2VCM?e=KBA6il?download=1) → 默认需放置根目录的models文件夹下
+ **额外缺少的文件**[→OneDrive](https://1drv.ms/u/s!AlYD8lJlPHCIh2xS1T_M_RBKkTIf?e=Wmgmtd?download=1) → 放置在```C:\Users\User name\.cache\torch\hub\checkpoints```

基础套件
---
+ Pyqt5 --> 界面窗口、按钮组、阈值选择、文件选择和进度条
+ Pytorch --> 追踪效果
+ Opencv --> 视频和摄像头追踪，播放与暂停

问题解答
---
有小伙伴提出dcnv2无法使用等问题。
对于此问题，本项目已提供了DCNv2的压缩包，解压后切换到DCNv2文件目录内执行```python setup.py install```,执行完后再将DCNv2文件夹设为根目录。
另外，对于库的路径问题，可在pycharm内将```FairMOT-main,src```文件夹设为根目录即可。

更新日志
---
2021.11.29  添加新分支ByteTrack-Kernel，以ByteTrack核心替换了当前的追踪核心
