import torch
import torch.nn as nn


class L2Norm(nn.Module):
    '''
    对conv4_3进行l2归一化
    '''

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))  # n_channels个随机数
        self.reset_parameters()

    def reset_parameters(self):
        # 使用gamma来填充weight的每个值
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        # 按通道进行求值
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps  # [1,1,38,38]
        x = torch.div(x, norm)
        # 将weight通过3个unsqueeze展开成[1,512,1,1],然后通过expand_as进行扩展,形成[1,512,38,38]
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


if __name__ == '__main__':
    input = torch.rand(1, 512, 38, 38)
    l2norm = L2Norm(512, 20)
    x = l2norm(input)
