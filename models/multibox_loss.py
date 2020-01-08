import torch
import torch.nn as nn
import torch.nn.functional as F
from data.config import voc
from models.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    '''
    SSD损失函数的计算
    '''

    def __init__(self, num_classes, overlap_thresh, neg_pos):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes  # 类别数
        self.threshold = overlap_thresh  # GT框与先验锚点框的阈值
        self.negpos_ratio = neg_pos  # 负例的比例

    def forward(self, predictions, targets):
        '''
        对损失函数进行计算:
            1.进行GT框与先验锚点框的匹配,得到loc_t和conf_t,分别表示锚点框需要匹配的坐标和锚点框需要匹配的label
            2.对包含目标的先验锚点框loc_t(即正例)与预测的loc_data计算位置损失函数
            3.对负例(即背景)进行损失计算,选择损失最大的num_neg个负例和正例共同组成训练样本,取出这些训练样本的锚点框targets_weighted
                与置信度预测值conf_p,计算置信度损失:
                a)为Hard Negative Mining计算最大置信度loss_c
                b)将loss_c中正例对应的值置0,即保留了所有负例
                c)对此loss_c进行排序,得到损失最大的idx_rank
                d)计算用于训练的负例的个数num_neg,约为正例的3倍
                e)选择idx_rank中前num_neg个用作训练
                f)将正例的index和负例的index共同组成用于计算损失的index,并从预测置信度conf_data和真实置信度conf_t提出这些样本,形成
                    conf_p和targets_weighted,计算两者的置信度损失.
        :param predictions: 一个元祖，包含位置预测，置信度预测，先验锚点框
                    位置预测：(batch_size,num_priors,4),即[batch_size,8732,4]
                    置信度预测：(batch_size,num_priors,num_classes),即[batch_size, 8732, 21]
                    先验锚点框：(num_priors,4),即[8732, 4]
        :param targets: 真实框的坐标与label，[batch_size,num_objs,5]
                    其中，5代表[xmin,ymin,xmia,ymax,label]
        '''
        loc_data, conf_data, priors = predictions
        num = loc_data.shape[0]  # 即batch_size大小
        priors = priors[:loc_data.shape[1], :]  # 取出8732个锚点框,与位置预测的锚点框数量相同
        num_priors = priors.shape[0]  # 8732

        loc_t = torch.Tensor(num, num_priors, 4)  # [batch_size,8732,4],生成随机tensor,后续用于填充
        conf_t = torch.Tensor(num, num_priors)  # [batch_size,8732]
        # 取消梯度更新,貌似默认是False
        loc_t.requires_grad = False
        conf_t.requires_grad = False

        for idx in range(num):
            truths = targets[idx][:, :-1]  # 坐标值,[xmin,ymin,xmia,ymax]
            labels = targets[idx][:, -1]  # label
            defaults = priors.cuda()
            match(self.threshold, truths, defaults, labels, loc_t, conf_t, idx)
        if torch.cuda.is_available():
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()  # shape:[batch_size,8732],其元素组成是类别标签号和背景

        pos = conf_t > 0  # 排除label=0,即排除背景,shape[batch_size,8732],其元素组成是true或者false
        # Localization Loss (Smooth L1),定位损失函数
        # Shape: [batch,num_priors,4]
        # pos.dim()表示pos有多少维,应该是一个定值(2)
        # pos由[batch_size,8732]变成[batch_size,8732,1],然后展开成[batch_size,8732,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)  # [num_pos,4],取出带目标的这些框
        loc_t = loc_t[pos_idx].view(-1, 4)  # [num_pos,4]
        # 位置损失函数
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')  # 这里对损失值是相加,有公式可知,还没到相除的地步

        # 为Hard Negative Mining计算max conf across batch
        batch_conf = conf_data.view(-1, self.num_classes)  # shape[batch_size*8732,21]
        # gather函数的作用是沿着定轴dim(1),按照Index(conf_t.view(-1, 1))取出元素
        # batch_conf.gather(1, conf_t.view(-1, 1))的shape[8732,1],作用是得到每个锚点框在匹配GT框后的label
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1).long())  # 这个不是最终的置信度损失函数

        # Hard Negative Mining
        # 由于正例与负例的数据不均衡,因此不是所有负例都用于训练
        loss_c[pos.view(-1, 1)] = 0  # pos与loss_c维度不一样,所以需要转换一下,选出负例
        loss_c = loss_c.view(num, -1)  # [batch_size,8732]
        _, loss_idx = loss_c.sort(1, descending=True)  # 得到降序排列的index
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos.sum(1, keepdim=True)  # pos里面是true或者false,因此sum后的结果应该是包含的目标数量
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)  # 生成一个随机数用于表示负例的数量,正例和负例的比例约3:1
        neg = idx_rank < num_neg.expand_as(idx_rank)  # [batch_size,8732] 选择num_neg个负例,其元素组成是true或者false

        # 置信度损失,包括正例和负例
        # [batch_size, 8732, 21],元素组成是true或者false,但true代表着存在目标,其对应的index为label
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # pos_idx由true和false组成,表示选择出来的正例,neg_idx同理
        # (pos_idx + neg_idx)表示选择出来用于训练的样例,包含正例和反例
        # torch.gt(other)函数的作用是逐个元素与other进行大小比较,大于则为true,否则为false
        # 因此conf_data[(pos_idx + neg_idx).gt(0)]得到了所有用于训练的样例
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted.long(), reduction='sum')

        # L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.sum()  # 一个batch里面所有正例的数量
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


if __name__ == '__main__':
    from mySSD.data.voc0712 import VOCDetection, detection_collate
    from mySSD.ssd import build_ssd

    train_data = VOCDetection('/home/dengshunge/Desktop/公司/公共数据集/VOC')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, collate_fn=detection_collate)
    batch_iterator = iter(train_loader)
    images, targets = next(batch_iterator)

    ssd_net = build_ssd('train')
    output = ssd_net(images)

    loss = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False)
    loss(output, targets)
