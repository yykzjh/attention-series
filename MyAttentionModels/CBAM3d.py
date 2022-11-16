# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/16 17:40
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.nn as nn




class CBAM3d(nn.Module):
    """
    "Convolutional Block Attention Module", reference to "http://arxiv.org/abs/1807.06521"
    """
    def __init__(self, channel, r=16, kernel_size=7):
        """
        定义一个卷积块注意力模块

        :param channel: 特征图的通道数
        :param r: 通道衰减系数
        :param kernel_size: 卷积核大小
        """
        super(CBAM3d, self).__init__()
        # 定义通道注意力模块
        self.channel_attention = ChannelAttention(channel=channel, r=r)
        # 定义空间注意力模块
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        # 拷贝输入数据，用于残差连接
        residual = x
        # 通过通道注意力模块
        out = x * self.channel_attention(x)
        # 通过空间注意力模块
        out = out * self.spatial_attention(out)

        # 返回残差连接的结果
        return out + residual



class ChannelAttention(nn.Module):
    """
    通道注意力
    """
    def __init__(self, channel, r=16):
        """
        定义通道注意力模块

        :param channel: 通道数
        :param r: 隐藏层衰减系数
        """
        super(ChannelAttention, self).__init__()
        # 定义全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 定义全局最大值池化
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        # 定义只有一个隐藏层的共享多层感知机(Shared MLP)
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(channel, channel // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // r, channel, 1, bias=False)
        )
        # 定义最后的激活函数Sigmoid
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 计算全局平均池化
        avg_result = self.avg_pool(x)
        # 计算全局最大值池化
        max_result = self.max_pool(x)
        # 分别将两类全局池化结果传入到共享感知机网络，得到两个通道描述符
        avg_out = self.shared_mlp(avg_result)
        max_out = self.shared_mlp(max_result)
        # 将两个通道描述符逐元素相加，然后将结果通过Sigmoid激活函数
        output = self.sigmoid(avg_out + max_out)

        return output



class SpatialAttention(nn.Module):
    """
    空间注意力
    """
    def __init__(self, kernel_size=7):
        """
        定义空间注意力模块

        :param kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()
        # 定义卷积层
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        # 定义最后的激活函数Sigmoid
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 求在通道轴上的平均池化
        avg_result = torch.mean(x, dim=1, keepdim=True)
        # 求在通道轴上的最大值池化
        max_result = torch.max(x, dim=1, keepdim=True)
        # 将两个特征平面拼接起来
        result = torch.cat([avg_result, max_result], dim=1)
        # 经过一个卷积层
        output = self.conv(result)
        # 通过最后的Sigmoid激活函数
        output = self.sigmoid(output)

        return output



