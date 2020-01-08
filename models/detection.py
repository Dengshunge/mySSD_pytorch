import torch
from torch.autograd import Function
from models.box_utils import decode, nms, nms
from data.config import voc


class Detect(Function):
    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh  # 非极大值抑制阈值
        self.conf_thresh = conf_thresh  # 置信度阈值

    def forward(self, loc_data, conf_data, prior_data):
        '''

        :param loc_data: 模型预测的锚点框位置偏差信息,shape[batch,num_priors,4]
        :param conf_data: 模型预测的锚点框置信度,[batch,num_priors,num_classes]
        :param prior_data: 先验锚点框,[num_priors,4]
        :return:最终预测结果,shape[batch,num_classes,top_k,5],其中5表示[置信度,xmin,ymin,xmax,ymax],
            top_k中前面不为0的是预测结果,后面为0是为了填充
        '''
        num = loc_data.shape[0]  # batch size
        num_priors = prior_data.shape[0]  # 8732
        output = torch.zeros(num, self.num_classes, self.top_k, 5)  # 保存结果
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)  # 置信度预测,transpose是为了后续操作方便

        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, voc['variance'])  # shape:[num_priors,4],对预测锚点框进行解码
            # 对每个类别,执行nms
            conf_scores = conf_preds[i].clone()  # shape:[num_classes,num_priors]
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # 和置信度阈值进行比较,大于为true,否则为false
                scores = conf_scores[cl][c_mask]  # 得到置信度大于阈值的那些锚点框置信度
                if scores.shape[0] == 0:
                    # 说明锚点框与这一类的GT框不匹配,简介说明,不存在这一类的目标
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)  # 得到置信度大于阈值的那些锚点框
                ids = nms(boxes, scores, self.nms_thresh, self.top_k)  # 对置信度大于阈值的那些锚点框进行nms,得到最终预测结果的index
                output[i, cl, :len(ids)] = torch.cat((scores[ids].unsqueeze(1),
                                                      boxes[ids]), 1)  # [置信度,xmin,ymin,xmax,ymax]
        return output


if __name__ == '__main__':
    from mySSD.data.voc0712 import VOCDetection, detection_collate
    from mySSD.ssd import build_ssd

    train_data = VOCDetection('/home/dengshunge/Desktop/公司/公共数据集/VOC')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, collate_fn=detection_collate)
    batch_iterator = iter(train_loader)
    images, targets = next(batch_iterator)

    ssd_net = build_ssd('test')
    output = ssd_net(images)

    detect = Detect(21, 200, 0.01, 0.45)
    detect(output[0], output[1], output[2])
