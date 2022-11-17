# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/17 14:32
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
from torch import nn
from collections import OrderedDict




class SKAttention3d(nn.Module):
    """
    选择性内核注意力, "Selective Kernel Attention"
    reference to "https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html"
    """
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], r=16, group=1, L=32):
        """
        定义一个选择性内核注意力模块

        :param channel: 输入通道数
        :param kernels: 不同分支的内核大小
        :param r: 通道数衰减系数
        :param group: 分组卷积的组数
        :param L: 通道数衰减最小值
        """
        super(SKAttention3d, self).__init__()
        # 计算通道数衰减后的数值
        self.d = max(channel // r, L)
        # 定义不同内核大小的分支
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, channel, kernel_size=k, padding=k//2, groups=group)),
                ("bn", nn.BatchNorm3d(channel)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义全连接层
        self.fc = nn.Linear(channel, self.d)
        # 定义各分支的通道注意力描述符
        self.fcs = nn.ModuleList([
            nn.Linear(self.d, channel)
            for i in range(len(kernels))
        ])
        # 定义应用于不同分支的Softmax
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):
        # 获得输入特征图维度信息
        bs, c, _, _, _ = x.size()

        # Split
        # 计算各分支不同内核大小的卷积结果
        conv_outs = [
            conv(x)
            for conv in self.convs
        ]

        # Fuse
        # 将各分支得到的特征图逐元素相加
        U = sum(conv_outs)  # bs, c, d, h, w
        # 计算全局平均池化
        S = U.mean(-1).mean(-1).mean(-1)  # bs, c
        # 计算紧凑特征Z
        Z = self.fc(S)  # bs, d


        # Select
        # 根据紧凑特征Z计算各分支的通道注意力描述符
        weights = [
            fc(Z).view(bs, c, 1, 1, 1)
            for fc in self.fcs
        ]
        # 堆叠各分支的通道注意力描述符
        attention_weights = torch.stack(weights, 0)  # k, bs, c, 1, 1, 1
        # 对同一个通道不同分支的注意力数值进行Softmax
        attention_weights = self.softmax(attention_weights)  # k, bs, c, 1, 1, 1
        # 将之前不同分支卷积得到的结果堆叠起来
        features = torch.stack(conv_outs, dim=0)  # k, bs, c, d, h, w

        # 根据注意力分数融合各分支的特征图
        V = torch.sum(attention_weights * features, dim=0)

        return V






if __name__ == '__main__':
    x = torch.randn((4, 256, 4, 4, 4))

    model = SKAttention3d(channel=256)

    output = model(x)

    print(x.size())
    print(output.size())












