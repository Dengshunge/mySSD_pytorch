import torch
import argparse
import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data.voc0712 import VOC_CLASSES as labelmap
from ssd import build_ssd
from data.voc0712 import VOCDetection, BaseTransform, VOCAnnotationTransform

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--trained_model', default='weights/VOC_final.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--voc_root', default='/home/dengshunge/Desktop/公司/公共数据集/VOC', type=str,
                    help='Location of VOC root directory')
parser.add_argument('--save_det_result', default='./det_result', type=str,
                    help='Location of saving detection result')
parser.add_argument('--cachedir', default='./cachedir', type=str,
                    help='Location of cache GT')
args = parser.parse_args()


def parse_rec(filename):
    # 类似与数据读取中的流程,读取相应的图片名和xml文件
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def write_voc_results_file(all_boxes, dataset):
    # 将检测结果按照每类写成文本,方便后面读取结果
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        if not os.path.exists(args.save_det_result):
            os.mkdir(args.save_det_result)
        filename = os.path.join(args.save_det_result, 'det_%s.txt' % (cls))
        with open(filename, 'w') as f:
            for im_ind, index in enumerate(dataset.ids):  # dataset.ids:[path,图片名]
                dets = all_boxes[cls_ind + 1][im_ind]  # 测试的时候,图片是按这个顺序读取的
                if dets == []:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(use_07=True):
    aps = []  # 保存所有类别的AP
    for i, cls in enumerate(labelmap):
        filename = os.path.join(args.save_det_result, 'det_%s.txt' % (cls))  # 读取这一类别的检测结果,对应上刚刚保存的结果
        rec, prec, ap = voc_eval(filename,
                                 os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', 'test.txt'),
                                 cls,
                                 args.cachedir,
                                 ovthresh=0.5,
                                 use_07_metric=use_07)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))


def voc_ap(rec, prec, use_07_metric=True):
    '''
    根据召回率和准确率,计算AP
    AP计算有两种方式:1.11点计算;2.最大面积计算
    :param rec: [1,num_all_detect]
    :param prec: [1,num_all_detect]
    '''
    if use_07_metric:
        # 旧版,11点计算
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            # 给召回率设定阈值,统计当召回率大于阈值的情况下,最大的准确率
            if np.sum(rec >= t) == 0:
                # 说明召回率没有比t更大
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11
    else:
        # 增加两个数字,是为了方便计算
        mrec = np.concatenate(([0.], rec, [1.]))  # shape:[1,num_all_detect+2]
        mpre = np.concatenate(([0.], prec, [0.]))

        # 计算最大面积
        for i in range(mpre.size - 1, 0, -1):
            # 取右边的最大值
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 得到与前面的数值不一样的index,可以理解成,计算面积时的边长
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 计算面积,长*宽
    return ap


def voc_eval(detpath,  # 某一类别下检测结果,每一行由文件名,置信度和检测坐标组成
             imagesetfile,  # 包含所有测试图片的文件
             classname,  # 需要检测的类别
             cachedir,  # 缓存GT框的pickle文件
             ovthresh=0.5,  # IOU阈值
             use_07_metric=True):
    '''
    假设检测结果在detpath.format(classname)下
    假设GT框坐标在annopath.format(imagename)
    假设imagesetfile每行仅包含一个文件名
    缓存所有GT框
    '''
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # 读取所有检测图片
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]  # 每张测试图片的名字

    # 下面代码是创建缓存文件,方便读取
    if not os.path.isfile(cachefile):
        # 不存在GT框缓存文件,则创建
        recs = {}  # key为图片名字,value为该图片下所有检测信息
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(
                os.path.join(args.voc_root, 'VOC2007', 'Annotations',
                             '%s.xml') % (imagename))  # 返回该图片下所有xml信息,包含所有目标
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # 保存下来,方便下次读取
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # 如果已经存在该文件,则加载回来即可
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # 为这一类提取GT框
    class_recs = {}
    npos = 0  # 这一类别的gt框总数
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]  # 提取某张测试图片下该类别的信息
        bbox = np.array([x['bbox'] for x in R])  # GT框坐标
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)  # 元素为true或者false,true表示难例
        det = [False] * len(R)  # 长度为len(R)的list,用于表示该GT框是否已经匹配过,len(R)可以理解为该测试图片下,该类别的数量
        npos = npos + sum(~difficult)  # 只选取非难例,计算非难例的个数,可以理解为GT框的个数
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    detfile = detpath.format(classname)  # 读取这一类的检测结果
    with open(detfile, 'r') as f:
        lines = f.readlines()

    if any(lines) == 1:  # 不为空
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]  # 图片名称集合,包含重复的
        confidence = np.array([float(x[1]) for x in splitlines])  # 置信度集合
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # 检测框集合

        # 根据置信度,降序排列
        sorted_ind = np.argsort(-confidence)  # 降序排名
        sorted_scores = np.sort(-confidence)  # 降序排列
        BB = BB[sorted_ind, :]  # 检测框根据置信度进行降序排列
        image_ids = [image_ids[x] for x in sorted_ind]

        nd = len(image_ids)  # 检测的目标总数
        tp = np.zeros(nd)  # 记录tp
        fp = np.zeros(nd)  # 记录fp,与tp互斥

        for d in range(nd):  # 循环每个预测框
            R = class_recs[image_ids[d]]  # 该图片下的真实信息
            bb = BB[d, :].astype(float)  # 预测框的坐标
            ovmax = -np.inf  # 预测框与GT框的IOU
            BBGT = R['bbox'].astype(float)  # GT框的坐标

            if BBGT.size > 0:
                # 计算多个GT框与一个预测框的IOU,选择最大IOU
                # 下面是计算IOU的流程
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)  # 得到该预测框与GT框最大的IOU值
                jmax = np.argmax(overlaps)  # 得到该预测框对应最大IOU的GT框的index

            if ovmax > ovthresh:
                # 当IOU大于阈值,才有机会判断为正例
                # 判断为fp有两种情况:
                # 1.该GT框被置信度高的预测框匹配过
                # 2.IOU小于阈值
                if not R['difficult'][jmax]:
                    # 该GT框要求之前没有匹配过
                    # 由于置信度是降序排序的,GT框只匹配置信度最高的,其余认为是FP
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1

        # 计算recall,precision
        fp = np.cumsum(fp)  # shape:[1,nd]
        tp = np.cumsum(tp)  # shape:[1,nd]
        rec = tp / float(npos)  # 召回率
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # 准确率,防止除0
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def evaluate_detections(all_boxes, dataset):
    write_voc_results_file(all_boxes, dataset)  # 将所有检测结果写成文本,保存下来
    do_python_eval(use_07=False)


if __name__ == '__main__':
    # 加载网络
    num_classes = len(labelmap) + 1
    net = build_ssd('test', 300, num_classes)
    net.load_state_dict(torch.load(args.trained_model))
    if torch.cuda.is_available():
        net = net.cuda()
        cudnn.benchmark = True
    net.eval()

    # 记载数据
    dataset = VOCDetection(args.voc_root, [('2007', 'test')], BaseTransform(300, (104, 117, 123)),
                           VOCAnnotationTransform())

    # 测试结果
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]  # 用于保存所有符合条件的结果
    with torch.no_grad():
        for i in range(num_images):
            img, gt, h, w = dataset.pull_item(i)
            img = img.unsqueeze(0)
            if torch.cuda.is_available():
                img = img.cuda()

            detections = net(img)  # 得到结果,shape[1,21,200,5]
            for j in range(1, detections.shape[1]):  # 循环计算每个类别
                dets = detections[0, j, :]  # shape[200,5],表示每个类别中最多200个锚点框,每个锚点框有5个值[conf,xmin,ymin,xmax,ymax]
                mask = dets[:, 0].gt(0.).expand(5, dets.shape[0]).t()  # 取出置信度大于0的情况.因为可能会出现实际有值的锚点框少于200个
                dets = torch.masked_select(dets, mask).view(-1, 5)  # 取出这些锚点框
                if dets.shape[0] == 0:
                    # 说明该图片不存在该类别
                    continue
                boxes = dets[:, 1:]  # 取出锚点框坐标
                # 计算出真实坐标
                boxes[:, 0] *= w
                boxes[:, 1] *= h
                boxes[:, 2] *= w
                boxes[:, 3] *= h

                scores = dets[:, 0].numpy()
                # np.newaxis增加一个新轴
                # 注意[xmin,ymin,xmax,ymax,conf]
                cls_dets = np.hstack((boxes.numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)

                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d}'.format(i + 1, num_images))  # 完成了一张图的所有类别检测

    print('Evaluating detections')
    evaluate_detections(all_boxes, dataset)
