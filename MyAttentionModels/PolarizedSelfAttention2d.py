# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/24 16:10
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.nn as nn
from torch.nn import functional as F



class ParallelPolarizedSelfAttention2d(nn.Module):
    """
    并行极化自注意力
    """
    def __init__(self, channel=512):
        """
        定义一个并行极化自注意力模块
        :param channel: 输入特征图的通道数
        """
        super(ParallelPolarizedSelfAttention2d, self).__init__()
        # 定义通道自注意的两个1X1卷积Wv、Wq
        self.ch_Wv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.ch_Wq = nn.Conv2d(channel, 1, kernel_size=1)
        # 定义通道注意力的Softmax
        self.ch_softmax = nn.Softmax(1)
        # 定义通道注意力对分数矩阵的卷积Wz
        self.ch_Wz = nn.Conv2d(channel // 2, channel, kernel_size=1)
        # 定义通道注意力的层归一化
        self.layer_norm = nn.LayerNorm((channel, 1, 1))
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义空间自注意的两个1X1卷积Wv、Wq
        self.sp_Wv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.sp_Wq = nn.Conv2d(channel, channel // 2, kernel_size=1)
        # 定义空间自注意力的全局自适应平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)


    def forward(self, x):
        # 获取输入特征图张量的维度信息
        bs, c, h, w = x.size()

        # Channel-only Self-Attention
        ch_Wv = self.ch_Wv(x)  # bs,c//2,h,w
        ch_Wq = self.ch_Wq(x)  # bs,1,h,w
        ch_Wv = ch_Wv.reshape(bs, c//2, -1)  # bs,c//2,h*w
        ch_Wq = ch_Wq.reshape(bs, -1, 1)  # bs,h*w,1
        ch_Wq = self.ch_softmax(ch_Wq)  # bs,h*w,1
        ch_Wz = torch.matmul(ch_Wv, ch_Wq).unsqueeze(-1)  # bs,c//2,1,1
        ch_score = self.sigmoid(self.layer_norm(self.ch_Wz(ch_Wz)))  # bs,c,1,1
        ch_out = ch_score * x

        # Spatial-only Self-Attention
        




















