# config.py
import os.path

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,  # 有20个类别，加上背景，总共21类
    'lr_steps': (80000, 100000, 120000),  # 学习率变化步数
    'max_iter': 120000,  # 最大迭代步数
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 特征图尺寸
    'min_dim': 300,  # 图片尺寸
    'steps': [8, 16, 32, 64, 100, 300],  # 特征图对应的缩放比
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # 宽高比
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
