import os
from data.augmentations import SSDAugmentation
import data.config as config
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch

# VOC的类别名称，总共20个
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform():
    '''
    获取xml里面的坐标值和label，并将坐标值转换成0到1
    '''

    def __init__(self, class_to_ind=None, keep_difficult=False):
        # 将类别名字转换成数字label
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        # 在xml里面，有个difficult的参数，这个表示特别难识别的目标，一般是小目标或者遮挡严重的目标
        # 因此，可以通过这个参数，忽略这些目标
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        '''
        将一张图里面包含若干个目标，获取这些目标的坐标值，并转换成0到1，并得到其label
        :param target: xml格式
        :return: 返回List,每个目标对应一行,每行包括5个参数[xmin, ymin, xmax, ymax, label_ind]
        '''
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1  # 判断该目标是否为难例
            # 判断是否跳过难例
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()  # text是获得目标的名称,lower将字符转换成小写,strip去除前后空格
            bbox = obj.find('bndbox')  # 获得真实框坐标

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1  # 获得坐标值
                # 将坐标转换成[0,1]，这样图片尺寸发生变化的时候，真实框也随之变化，即平移不变形
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]  # 获得名字对应的label
            bndbox.append(label_idx)
            res.append(bndbox)  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection():
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the input image
            图片预处理的方式，这里使用了大量数据增强的方式
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
            真实框预处理的方式
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=SSDAugmentation(size=config.voc['min_dim'], mean=config.MEANS),
                 target_transform=VOCAnnotationTransform()):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = []
        # 使用VOC2007和VOC2012的train作为训练集
        for (year, name) in self.image_set:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append([rootpath, line[:-1]])

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt, h, w

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = tuple(self.ids[index])
        # img_id里面有2个值
        target = ET.parse(self._annopath % img_id).getroot()  # 获得xml的内容，但这个是具有特殊格式的
        img = cv2.imread(self._imgpath % img_id)
        height, width, _ = img.shape

        if self.target_transform is not None:
            # 真实框处理
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            # 图像预处理，进行数据增强,只在训练进行数据增强,测试的时候不需要
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # 转换格式
            img = img[:, :, (2, 1, 0)]  # to rbg
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''
        测试时候用,返回opencv读取的图像
        '''
        img_id = tuple(self.ids[index])
        return cv2.imread(self._imgpath % img_id)

    def pull_anno(self, index):
        '''
        测试时候用,返回图片名称和原始未经缩放的GT框
        :param index:
        :return:
        '''
        img_id = tuple(self.ids[index])
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    自定义处理在同一个batch,含有不同数量的目标框的情况

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        image -= self.mean
        image = image.astype(np.float32)
        return image, boxes, labels


if __name__ == '__main__':
    # target_transform = VOCAnnotationTransform()
    # transform = SSDAugmentation(size=config.voc['min_dim'], mean=config.MEANS)
    # path = '/home/dengshunge/Desktop/公司/公共数据集/VOC/VOC2007/Annotations/006346.xml'
    # target = ET.parse(path).getroot()
    # img = cv2.imread('/home/dengshunge/Desktop/公司/公共数据集/VOC/VOC2007/JPEGImages/006346.jpg')
    #
    # height, width, _ = img.shape
    #
    # target = target_transform(target, height, width)
    #
    # target = np.array(target)
    #
    #
    # img, boxes, labels = transform(img, target[:, :4], target[:, :4])
    import torch

    data = VOCDetection('/home/dengshunge/Desktop/公司/公共数据集/VOC')
    loader = torch.utils.data.DataLoader(data, batch_size=1, collate_fn=detection_collate)
    batch_iterator = iter(loader)
    for i in range(10):
        x, y = next(batch_iterator)
        print(y)
