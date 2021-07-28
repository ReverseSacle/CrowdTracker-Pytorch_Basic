制作介绍
===

思路
---

+ **Detection and Re_ID**

<img width="707" alt="FairMOT" src="https://user-images.githubusercontent.com/73418195/122189832-d73d5e80-cec3-11eb-8b32-95ef7c8ea3b1.png">

该网络主要为目标检测和重识别一体化。
对于one-stage跟踪都是基于anchor锚，这造成了提取的特征未与对象中心对齐，例如当两个目标相互靠近时，ahchor的位置就不太准确了。

图中，用点代表目标来提高位置的准确性。此外，与以往的通过高维特征来Re_ID相比，低维特征对MOT更好，因为它的训练图像比ReID少。学习低维特征有助于减少过拟合小数据的风险，并提高跟踪的稳定性。

+ [Paper_地址](https://arxiv.org/abs/2004.01888)
