import torch
import argparse
from data.voc0712 import VOCDetection, detection_collate
import torch.optim as optim
import torch.backends.cudnn as cudnn
from ssd import build_ssd
from data.config import voc
from models.multibox_loss import MultiBoxLoss
from tqdm import tqdm


# 定义模型中需要使用的参数
def get_args():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument('--dataset', default='VOC', type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default='/home/dengshunge/Desktop/公司/公共数据集/VOC', type=str,
                        help='Dataset root directory path')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')

    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    args = parser.parse_args()
    return args


def main():
    '''
    1.读取默认参数
    2.读取数据
    3.创建网络
    4.设置优化器、损失函数
    5.进入train函数或者eval函数
    '''
    args = get_args()

    # 读取数据
    if args.dataset == 'VOC':
        train_dataset = VOCDetection(root=args.dataset_root)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=4,
            collate_fn=detection_collate, shuffle=True, pin_memory=True)
    # create batch iterator
    batch_iterator = iter(train_loader)

    # 创建网络
    ssd_net = build_ssd('train', voc['min_dim'], voc['num_classes'])

    if torch.cuda.is_available():
        ssd_net = ssd_net.cuda()
        cudnn.benchmark = True

    # 优化器SGD
    optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # 损失函数
    criterion = MultiBoxLoss(num_classes=voc['num_classes'],
                             overlap_thresh=0.5,
                             neg_pos=3)

    # 进入训练
    ssd_net.train()

    start_iter = 0  # 训练步数
    step_index = 0  # 统计学习率
    for iteration in tqdm(range(start_iter, voc['max_iter'])):
        # 调整学习率
        if iteration in voc['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args, step_index)

        # 加载图像
        try:
            images, targets, _, _ = next(batch_iterator)
        except:
            batch_iterator = iter(torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, num_workers=4,
                collate_fn=detection_collate, shuffle=True, pin_memory=True))
            images, targets, _, _ = next(batch_iterator)

        images = images.cuda()
        targets = [x.cuda() for x in targets]

        # 前向
        out = ssd_net(images)
        # 后向
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)

        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()))

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_VOC_' + repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), 'weights/VOC_final.pth')


def adjust_learning_rate(optimizer, args, step):
    lr = args.lr * (args.gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
