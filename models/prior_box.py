import torch
from itertools import product as product
from math import sqrt as sqrt


def PriorBox(cfg):
    '''
    为所有特征图生成预设的锚点框，返回所有生成的锚点框，尺寸为[8732,4]，
    每行表示[中心点x，中心点y，宽，高]
    '''
    image_size = cfg['min_dim']  # 300
    feature_maps = cfg['feature_maps']  # [38, 19, 10, 5, 3, 1],特征图尺寸
    steps = cfg['steps']  # [8, 16, 32, 64, 100, 300]
    min_sizes = cfg['min_sizes']  # [30, 60, 111, 162, 213, 264]
    max_sizes = cfg['max_sizes']  # [60, 111, 162, 213, 264, 315]
    aspect_ratios = cfg['aspect_ratios']  # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    mean = []
    # 为所有特征图生成锚点框
    for k, f in enumerate(feature_maps):
        # product(list1,list2)的作用是依次取出list1中的每1个元素，与list2中的每1个元素，
        # 组成元组，然后，将所有的元组组成一个列表，返回
        # 而这里使用了repeat,说明1个list重复2次
        for i, j in product(range(f), repeat=2):
            f_k = image_size / steps[k]
            # 计算中心点,这里的j是沿x方向变化的
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            # aspect_ratio=1有两种情况,s_k=s_k,s_k=sqrt(s_k*s_(k+1))
            s_k = min_sizes[k] / image_size
            mean += [cx, cy, s_k, s_k]

            s_k_prime = sqrt(s_k * (max_sizes[k] / image_size))
            mean += [cx, cy, s_k_prime, s_k_prime]

            # 剩余的aspect_ratio
            for ar in aspect_ratios[k]:
                mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

    # 此时的mean是1*34928的list，要4个数就分割出来，所以需要用view,从而变成[8732,4],即有8732个锚点框
    output = torch.Tensor(mean).view(-1, 4)
    if cfg['clip']:
        # 对每个元素进行截断限制,限制为[0,1]之间
        output.clamp_(min=0, max=1)
    return output


if __name__ == '__main__':
    import sys

    sys.path.append('..')
    import data.config as config

    priorbox = PriorBox(config.voc)
