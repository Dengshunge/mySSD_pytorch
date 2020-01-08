import torch
from data.config import voc

def point_from(boxes):
    '''
    将先验锚点框(cx,cy,w,h)转换成(xmin,ymin,xmax,ymax)
    :param boxes: 先验锚点框的坐标
    :return: 返回坐标转换后的先验锚点框(xmin,ymin,xmax,ymax)
    '''
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xim,ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax,ymax


def intersect(box_a, box_b):
    '''
    计算box_a和box_b两种框的交集，两种框的数量可能不一致.
    首先将框resize成[A,B,2],这样做的目的是,能计算box_a中每个框与
    box_b中每个框的交集:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
    然后再计算框的交集
    :param box_a:[A,4],4代表[xmin,ymin,xmax,ymax]
    :param box_b:[B,4],4代表[xmin,ymin,xmax,ymax]
    :return:交集面积,shape[A,B]
    '''
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)  # 得到x与y之间的差值，shape为[A,B,2]
    return inter[:, :, 0] * inter[:, :, 1]  # 按位相乘,shape[A,B],每个元素为矩形框的交集面积


def jaccard(box_a, box_b):
    '''
    计算box_a中每个矩形框与box_b中每个矩形框的IOU
    :param box_a:真实框的坐标,Shape: [num_objects,4]
    :param box_b:先验锚点框的坐标,Shape: [num_priors,4]
    :return:IOU,Shape: [box_a.size(0), box_b.size(0)]
    '''
    inter = intersect(box_a, box_b)  # 计算交集面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter  # 并集面积
    return inter / union  # shape[A,B]


def match(threshold, truths, priors, labels, loc_t, conf_t, idx):
    '''
    这个函数对应论文中的matching strategy匹配策略.SSD需要为每一个先验锚点框都指定一个label,
    这个label或者指向背景,或者指向每个类别.
    论文中的匹配策略是:
        1.首先,每个GT框选择与其IOU最大的一个锚点框,并令这个锚点框的label等于这个GT框的label
        2.然后,当锚点框与GT框的IOU大于阈值(0.5)时,同样令这个锚点框的label等于这个GT框的label
    因此,代码上的逻辑为:
        1.计算每个GT框与每个锚点框的IOU,得到一个shape为[num_object,num_priors]的矩阵overlaps
        2.选择与GT框的IOU最大的锚点框,锚点框的index为best_prior_idx,对应的IOU值为best_prior_overlap
        3.为每一个锚点框选择一个IOU最大的GT框,可能会出现多个锚点框匹配一个GT框的情况,此时,每个锚点框对应GT框的index为best_truth_idx,
            对应的IOU为best_truth_overlap.注意,此时IOU值可能会存在小于阈值的情况.
        4.第3步可能到导致存在GT框没有与锚点框匹配上的情况,所以要和第2步进行结合.在第3步的基础上,对best_truth_overlap进行选择,选择出
            best_prior_idx这些锚点框,让其对其的IOU等于一个大于1的定值;并且让best_truth_idx中index为best_prior_idx的锚点框的label
            与GT框对应上.最终,best_truth_overlap表示每个锚点框与GT框的最大IOU值,而best_truth_idx表示每个锚点框用于与相应的GT框进行
            匹配.
        5.第4步中,会存在IOU小于阈值的情况,要将这些小于IOU阈值的锚点框的label指向背景,完成第二条匹配策略.
            labels表示GT框对应的标签号,"conf=labels[best_truth_idx]+1"得到每个锚点框对应的标签号,其中label=0是背景.
            "conf[best_truth_overlap < threshold] = 0"则将小于IOU阈值的锚点框的label指向背景
        6.得到的conf表示每个锚点框对应的label,还需要一个矩阵,来表示每个锚点框需要匹配GT框的坐标.
            truths表示GT框的坐标,"matches = truths[best_truth_idx]"得到每个锚点框需要匹配GT框的坐标.
    :param threshold:IOU的阈值
    :param truths:GT框的坐标,shape:[num_obj,4]
    :param priors:先验锚点框的坐标,shape:[num_priors,4],num_priors=8732
    :param labels:这些GT框对应的label,shape:[num_obj],此时label=0还不是背景
    :param loc_t:坐标结果会保存在这个tensor
    :param conf_t:置信度结果会保存在这个tensor
    :param idx:结果保存的idx
    '''
    # 第1步,计算IOU
    overlaps = jaccard(truths, point_from(priors))  # shape:[num_object,num_priors]

    # 第2步,为每个真实框匹配一个IOU最大的锚点框,GT框->锚点框
    # best_prior_overlap为每个真实框的最大IOU值,shape[num_objects,1]
    # best_prior_idx为对应的最大IOU的先验锚点框的Index,其元素值的范围为[0,num_priors]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # 第3步,若先验锚点框与GT框的IOU>阈值,也将这些锚点框匹配上,锚点框->GT框
    # best_truth_overlap为每个先验锚点框对应其中一个真实框的最大IOU,shape[1,num_priors]
    # best_truth_idx为每个先验锚点框对应的真实框的index,其元素值的范围为[0,num_objects]
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_prior_idx.squeeze_(1)  # [num_objects]
    best_prior_overlap.squeeze_(1)  # [num_objects]
    best_truth_idx.squeeze_(0)  # [num_priors],8732
    best_truth_overlap.squeeze_(0)  # [num_priors],8732

    # 第4步
    # index_fill_(self, dim: _int, index: Tensor, value: Number)对第dim行的index使用value进行填充
    # best_truth_overlap为第一步匹配的结果,需要使用到,使用best_prior_idx是第二步的结果,也是需要使用上的
    # 所以在best_truth_overlap上进行填充,表明选出来的正例
    # 使用2进行填充,是因为,IOU值的范围是[0,1],只要使用大于1的值填充,就表明肯定能被选出来
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # 确定最佳先验锚点框
    # 确保每个GT框都能匹配上最大IOU的先验锚点框
    # 得到每个先验锚点框都能有一个匹配上的数字
    # best_prior_idx的元素值的范围是[0,num_priors],长度为num_objects
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # 第5步
    conf = labels[best_truth_idx] + 1  # Shape: [num_priors],0为背景,所以其余编号+1
    conf[best_truth_overlap < threshold] = 0  # 置信度小于阈值的label设置为0

    # 第6步
    matches = truths[best_truth_idx]  # 取出最佳匹配的GT框,Shape: [num_priors,4]

    # 进行位置编码
    loc = encode(matches, priors,voc['variance'])
    loc_t[idx] = loc  # [num_priors,4],应该学习的编码偏差
    conf_t[idx] = conf  # [num_priors],每个锚点框的label


def encode(matched, priors, variances):
    '''
    对坐标进行编码,对应论文中的公式2
    利用GT框和先验锚点框,计算偏差,用于回归
    :param matched: 每个先验锚点框对应最佳的GT框,Shape: [num_priors, 4],
                    其中4代表[xmin,ymin,xmax,ymax]
    :param priors: 先验锚点框,Shape: [num_priors,4],
                    其中4代表[中心点x,中心点y,宽,高]
    :return: shape:[num_priors, 4]
    '''
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]  # 计算GT框与锚点框中心点的距离
    g_cxcy /= (variances[0] * priors[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2])  # xmax-xmin,ymax-ymin
    g_wh /= priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], 1)


def decode(loc, priors, variances):
    '''
    对编码的坐标进行解码,返回预测框的坐标
    :param loc: 网络预测的锚点框偏差信息,shape[num_priors,4]
    :param priors: 先验锚点框,[num_priors,4]
    :return: 预测框的坐标[num_priors,4],4代表[xmin,ymin,xmax,ymax]
    '''
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * priors[:, 2:] * variances[0],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)  # [中心点x,中心点y,宽,高]
    # boxes = torch.cat((
    #     priors[:, :2] + loc[:, :2] * priors[:, 2:],
    #     priors[:, 2:] * torch.exp(loc[:, 2:])), 1)  # [中心点x,中心点y,宽,高]
    boxes[:, :2] -= boxes[:, 2:] / 2  # xmin,ymin
    boxes[:, 2:] += boxes[:, :2]  # xmax,ymax
    return boxes


def log_sum_exp(x):
    '''
    用于计算在batch所有样本突出的置信度损失
    :param x: 置信度网络预测出来的置信度
    :return:
    '''
    x_max = x.max()  # 得到x中最大的一个数字
    # torch.sum(torch.exp(x - x_max), 1, keepdim=True)->shape:[8732,1],按行相加
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def nms(boxes, scores, overlap=0.5, top_k=200):
    '''
    进行nms操作
    :param boxes: 模型预测的锚点框的坐标
    :param scores: 模型预测的锚点框对应某一类的置信度
    :param overlap:nms阈值
    :param top_k:选取前top_k个预测框进行操作
    :return:预测框的index
    '''
    keep = torch.zeros(scores.shape[0])
    if boxes.numel() == 0:  # numel()返回tensor里面所有元素的个数
        return keep
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]  # xmin,ymin,xmax,ymax
    area = torch.mul(x2 - x1, y2 - y1)
    _, idx = scores.sort(0)  # 升序排序
    idx = idx[-top_k:]  # 取得最大的top_k个置信度对应的index

    keep = []  # 记录最大最终锚点框的index
    while idx.numel() > 0:
        i = idx[-1]  # 取出置信度最大的锚点框的index
        keep.append(i)
        idx = idx[:-1]
        if idx.numel() == 0:
            break
        IOU = jaccard(boxes[i].unsqueeze(0), boxes[idx])  # 计算这个锚点框与其余锚点框的iou
        mask = IOU.le(overlap).squeeze(0)
        idx = idx[mask]  # 排除大于阈值的锚点框
    return torch.tensor(keep)


if __name__ == '__main__':
    from mySSD.data.voc0712 import VOCDetection, detection_collate
    from mySSD.models.prior_box import PriorBox
    from mySSD.data.config import voc

    train_data = VOCDetection('/home/dengshunge/Desktop/公司/公共数据集/VOC')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, collate_fn=detection_collate)
    batch_iterator = iter(train_loader)
    img, label = next(batch_iterator)

    priors = torch.Tensor(PriorBox(voc)).data
    truths = label[0][:, :-1]
    labels = label[0][:, -1]
    loc_t = torch.Tensor(truths.shape[0], priors.shape[0], 4)
    conf_t = torch.LongTensor(truths.shape[0], priors.shape[0])

    print('truths: ', truths)
    print('labels: ', labels)
    match(0.5, truths, priors, [0.1, 0.2], labels, loc_t, conf_t, 0)
