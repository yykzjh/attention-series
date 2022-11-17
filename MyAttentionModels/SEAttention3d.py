# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/17 22:35
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.nn as nn




class SEAttention3d(nn.Module):
    """
    压缩与激励注意力, "Squeeze-and-Excitation"
    reference to "DOI:10.1109/CVPR.2018.00745"
    """
    def __init__(self, channel=512, r=16):
        """
        定义压缩与激励通道注意力模块

        :param channel: 输入图像的通道数
        :param r: 全连接中间过程的通道数衰减系数
        """
        super(SEAttention3d, self).__init__()
        # 定义一个全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 定义激励部分的全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        # 获取维度信息
        b, c, _, _, _ = x.size()
        # Squeeze,即通过全局平均池化压缩空间信息
        y = self.avg_pool(x).view(b, c)
        # 通过全连接层捕获通道依赖性
        y = self.fc(y).view(b, c, 1, 1, 1)

        # 将通道注意力权重与输入特征图相乘，增强重要信息，抑制不重要的信息
        out = x * y

        return out








if __name__ == '__main__':
    x = torch.rand((4, 256, 4, 4, 4))

    model = SEAttention3d(channel=256)

    output = model(x)

    print(x.size())
    print(output.size())

