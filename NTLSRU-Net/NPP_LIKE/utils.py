import torch
import numpy as np
import matplotlib.pyplot as plt

'''神经网络权重初始化'''
def weights_init(m, weight_init_name, config):
    if weight_init_name == 'normal':
        return weights_init_normal(m, config.weight_init_normal_mean, config.weight_init_normal_std)
    elif weight_init_name == 'kaming':
        return weights_init_kaming(m)
    elif weight_init_name == 'xavier':
        weigth_init_xavier(m)

def weights_init_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

def weights_init_kaming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in',# 对于全连接层，fan_in是输入维度，fan_out是输出维度；对于卷积层，设其维度为[Cout,Cin,H,W],其中H*W为kernel规模。则fan_in是H*W*Cin,fan_out是H*W*Cout
                                nonlinearity='relu')  # kaiming_normal_:何凯明提出了针对于relu的初始化方法。pytorch默认使用kaiming正态分布初始化卷积层参数。
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

def weigth_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()

def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def clip_img(data,padding):
    if len(data.shape) == 2:
        h,w = data.shape
        return data[padding:h-padding,padding:w-padding]
    elif len(data.shape) == 3:
        c,h,w = data.shape
        return data[:,padding:h - padding, padding:w - padding]
    elif len(data.shape) == 4:
        b,c,h,w = data.shape
        return data[:,:, padding:h - padding, padding:w - padding]






