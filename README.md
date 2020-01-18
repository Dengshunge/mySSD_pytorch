# mySSD_pytorch
本project使用pytorch来实现最简单的SSD代码，主要参考了[这个GITHUB](https://github.com/amdegroot/ssd.pytorch)，包括从数据的读取、处理，模型网络的搭建、损失函数的构建和模型的测试。用最简单的代码来学习到SSD中最重要的部分:
- train.py包含了搭建这个网络的主体部分
- ssd.py包含了构建ssd网络结构
- eval.py是用于评估ssd网络
- models文件夹包含了用于构建网络的常用模块，例如bounding box相关函数，检测函数，L2正则化，损失函数，先验框函数。
- data文件夹包含了对数据的相关处理，主要是VOC的，含有数据增强。

同时提供一下原有的已经训练好的[pytorch models](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth)，通过加载这份models，mAP可以达到77.43%.
## Train
1. 修改train.py中相关的路径
2. 运行train.py
```
python3 train.py
```

## Eval
1. 修改eval.py中相关路径和pth路径
2. 运行eval.py

```
python3 eval.py
```

