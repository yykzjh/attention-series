# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/18 17:44
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.nn as nn



class GridAttentionGate2d(nn.Module):
    """
    网格注意力门控模块
    reference to "http://arxiv.org/abs/1804.03999"
    """
    def __init__(self, F_l, F_g, F_int=None, mode="concatenation", sub_sample_factor=2):
        """
        定义一个网格注意力门控模块

        :param F_l: 输入特征图的通道数(一般是跳跃连接的特征图通道数)
        :param F_g: 门控特征图的通道数(一般是上采样前的特征图的通道数)
        :param F_int: 中间层特征图的通道数(一般是输入特征图通道数的一半)
        :param mode: 前向传播计算模式
        :param sub_sample_factor: 上层和下层特征图的尺寸比例
        """
        super(GridAttentionGate2d, self).__init__()
        # 定义中间层特征图的通道数
        if F_int is None:
            F_int = F_l // 2
            if F_int == 0:
                F_int = 1
        # 最终结果输出前的一个点卷积
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=F_l, out_channels=F_l, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_l)
        )

        # 定义输入信号的门控信号的结合部分,Theta^T * x_ij + Phi^T * gating_signal + bias
        # 定义输入特征图的变换卷积，输入特征图输入上层，尺寸较大，需要下采样
        self.theta = nn.Conv2d(in_channels=F_l, out_channels=F_int, kernel_size=sub_sample_factor,
                               stride=sub_sample_factor, padding=0, bias=False)
        # 定义门控特征图的变换卷积,bias=True等于公式中最后加的bias
        self.phi = nn.Conv2d(in_channels=F_g, out_channels=F_int, kernel_size=1, stride=1, padding=0, bias=True)

        # 定义ψ，将结合后的特征图通道数降为1
        self.psi = nn.Conv2d(in_channels=F_int, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        # 定义计算注意力分数的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 根据指定的模式选择不同的函数执行前向传播操作
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('未知的操作函数！')


    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output


    def _concatenation(self, x, g):
        pass


    def _concatenation_debug(self, x, g):
        pass


    def _concatenation_residual(self, x, g):
        pass














