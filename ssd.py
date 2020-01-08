import torch
import torch.nn as nn
import torch.nn.functional as F
from data.config import voc
import numpy as np
from models.l2norm import L2Norm
from models.prior_box import PriorBox
from models.detection import Detect

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
            512, 512, 512],  # M表示maxpolling
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],  # S表示stride=2
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # 每个特征图的每个点，对应锚点框的数量
    '512': [],
}


class SSD(nn.Module):
    '''
    构建SSD的主函数，将base(vgg)、新增网络和位置网络与置信度网络组合起来
    '''

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priors = torch.Tensor(PriorBox(voc))
        self.size = size

        # SSD网络
        self.vgg = nn.ModuleList(base)
        # 对conv4_3的特征图进行L2正则化
        self.L2Norm = L2Norm(512, 20)

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes=self.num_classes, top_k=200,
                                 conf_thresh=0.01, nms_thresh=0.45)

    def forward(self, x):
        sources = []  # 保存特征图
        loc = []  # 保存每个特征图进行位置网络后的信息
        conf = []  # 保存每个特征图进行置信度网络后的信息

        # 处理输入至conv4_3
        for k in range(23):
            x = self.vgg[k](x)

        # 对conv4_3进行L2正则化
        s = self.L2Norm(x)
        sources.append(s)

        # 完成vgg后续的处理
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # 使用新增网络进行处理
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # 将特征图送入位置网络和置信度网络
        # l(x)或者c(x)的shape为[batch_size,channel,height,weight],使用了permute后,变成[batch_size,height,weight,channel]
        # 这样做应该是为了方便后续处理
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 进行格式变换
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # [batch_size,34928],锚点框的数量8732*4=34928
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'train':
            output = (loc.view(loc.size(0), -1, 4),  # [batch_size,num_priors,4]
                      conf.view(conf.size(0), -1, self.num_classes),  # [batch_size,num_priors,21]
                      self.priors)  # [num_priors,4]
        else:  # Test
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # 位置预测
                self.softmax(conf.view((conf.size(0), -1, self.num_classes))),  # 置信度预测
                self.priors.cuda()  # 先验锚点框
            )

        return output


def vgg(cfg=base['300'], batch_norm=False):
    '''
    该函数来源于torchvision.models.vgg19()中的make_layers()
    '''
    layers = []
    in_channels = 3

    # vgg主体部分,到论文的conv6之前
    for v in cfg:
        if v == 'M':
            # ceil_mode是向上取整
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)

    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers


def add_extras(cfg=extras['300'], in_channels=1024):
    '''
    完成SSD后半部分的网络构建，即作者新加上去的网络,从conv7之后到conv11_2
    '''
    layers = []
    flag = False  # 交替控制卷积核,使用1*1或者使用3*3
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_class):
    '''
    返回vgg网络,新增网络,位置网络和置信度网络
    '''
    loc_layers = []  # 判断位置
    conf_layers = []  # 判断置信度

    vgg_source = [21, -2]  # 21表示conv4_3的序号,-2表示conv7的序号
    for k, v in enumerate(vgg_source):
        # vgg[v]表示需要提取的特征图
        # cfg[k]代表该特征图下每个点对应的锚点框数量
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_class, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        # [1::2]表示,从第1位开始，步长为2
        # 这么做的目的是，新增加的层，都包含2层卷积，需要提取后面那层卷积的结果作为特征图
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_class, kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers)


def build_ssd(phase, size=300, num_class=21):
    if phase != 'test' and phase != 'train':
        raise ("ERROR: Phase: " + phase + " not recognized")
    base_, extras_, head_ = multibox(vgg(base[str(size)]),
                                     add_extras(extras[str(size)], in_channels=1024),
                                     mbox[str(size)],
                                     num_class)
    return SSD(phase, size, base_, extras_, head_, num_class)


if __name__ == '__main__':
    img = torch.rand(1, 3, 300, 300)

    ssd_net = build_ssd('train')
    output = ssd_net(img)
