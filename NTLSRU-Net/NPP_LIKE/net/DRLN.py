import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from NPP_LIKE.base_networks import ConvBlock,Upsample2xBlock

class LaplacianAttentionBlock(nn.Module):
    def __init__(self,cin, reduction=16):
        super(LaplacianAttentionBlock,self).__init__()
        self.cin = cin
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1 = ConvBlock(self.cin,self.cin//reduction,kernel_size=3,stride=1,padding=3,dilation=3,activation='relu',norm=None)
        self.c2 = ConvBlock(self.cin, self.cin // reduction, kernel_size=3, stride=1, padding=5, dilation=5,activation='relu',norm=None)
        self.c3 = ConvBlock(self.cin, self.cin // reduction, kernel_size=3, stride=1, padding=7, dilation=7,activation='relu',norm=None)
        self.c4 = ConvBlock((self.cin // reduction)*3,self.cin, kernel_size=3, stride=1, padding=1,activation='sigmoid',norm=None)
    def forward(self,x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y


class ResBlock(nn.Module):
    def __init__(self,cin,cout):
        super(ResBlock,self).__init__()
        self.cin = cin
        self.cout = cout
        self.conv1 = ConvBlock(cin,cout,kernel_size=3,stride=1,padding=1,activation='relu',norm=None)
        self.conv2 = ConvBlock(cout,cout,kernel_size=3,stride=1,padding=1,activation=None,norm=None)
    def forward(self,x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.add(res,out)
        return out

class DenseResBlock(nn.Module):
    def __init__(self,cin,cout):
        super(DenseResBlock,self).__init__()
        self.cin = cin
        self.cout = cout
        self.r1 = ResBlock(cin,cin)
        self.r2 = ResBlock(cin*2,cin*2)
        self.r3 = ResBlock(cin*4,cin*4)
        self.g = ConvBlock(cin*8,cout,kernel_size=1,stride=1,padding=0,activation='relu',norm=None)
    def forward(self,x):
        c0 = x
        r1 = self.r1(c0)
        c1 = torch.cat([c0,r1],dim=1)

        r2 = self.r2(c1)
        c2 = torch.cat([c1,r2],dim=1)

        r3 = self.r3(c2)
        c3 = torch.cat([c2,r3],dim=1)

        out = self.g(c3)
        return out

class DenseResAttentionBlock(nn.Module):
    def __init__(self,cin,cout,attentiontype='LCA'):
        super(DenseResAttentionBlock,self).__init__()
        self.attentiontype = attentiontype
        self.DRB = DenseResBlock(cin,cout)
        if self.attentiontype == 'LCA':
            self.atten = LaplacianAttentionBlock(cout)

    def forward(self,x):
        out = self.DRB(x)
        if self.attentiontype is not None:
            out = self.atten(out)
        return out

class MSC(nn.Module):
    '''Medium Skip Connection'''
    def __init__(self,base_filter,attentiontype='LCA',nums_DRAblk=2):
        super(MSC,self).__init__()
        self.nums_DRAblk = nums_DRAblk
        DRA_blocks = []
        conv_lst = []
        for i in range(nums_DRAblk):
            DRA_blocks.append(DenseResAttentionBlock(base_filter,base_filter,attentiontype))
            conv_lst.append(ConvBlock(base_filter*(i+2),base_filter,kernel_size=3,stride=1,padding=1,activation='relu',norm=None))
        self.DRA_layers = nn.Sequential(*DRA_blocks)
        self.DRA_Cov_layers = nn.Sequential(*conv_lst)
    def forward(self,x):
        res = x
        ci = oi = x
        for i in range(self.nums_DRAblk):
            bi = self.DRA_layers[i](oi)
            ci = torch.cat([ci,bi],dim=1)
            oi = self.DRA_Cov_layers[i](ci)
        out = torch.add(res,oi)
        return out


class DNLFeatureExtract(nn.Module):
    '''
    DNL输入网络前期处理：特征提取
    '''
    def __init__(self,num_channels, base_filter=64):
        super(DNLFeatureExtract,self).__init__()
        self.num_channels = num_channels
        self.base_filter = base_filter
        self.conv1 = ConvBlock(num_channels,base_filter,3,1,1,activation=None,norm=None)
    def forward(self, x):
        out = self.conv1(x)
        return out

class DNLReconstruction(nn.Module):
    '''
    DNL重建的最后一部分网络：图像重建
    '''
    def __init__(self,base_filter=64,upsample='ps'):
        super(DNLReconstruction,self).__init__()
        self.base_filter = base_filter
        self.upsample = upsample
        self.conv1 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None, norm=None)
        self.conv2 = ConvBlock(base_filter, 1, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.add(out,x)
        out = self.conv2(out)
        return out

class DNLNonlinearMapping(nn.Module):
    def __init__(self,base_filter=64,nums_msc=3,attentiontype='LCA',nums_DRAblk=2):
        super(DNLNonlinearMapping,self).__init__()
        self.nums_msc = nums_msc
        msc_blks = []
        for i in range(nums_msc):
            msc_blks.append(MSC(base_filter,attentiontype,nums_DRAblk))
        self.msc_layers = nn.Sequential(*msc_blks)
    def forward(self, x):
        res = x
        out = self.msc_layers(x)
        out = torch.add(res,out)
        return out

class DNLSRNet(nn.Module):
    def __init__(self,config,cin,base_filter=64):
        super(DNLSRNet,self).__init__()
        self.upsample = config.upsample
        self.nums_msc = config.nums_msc
        self.attentiontype = config.attentiontype
        self.nums_DRAblk = config.nums_DRAblk
        self.head = DNLFeatureExtract(cin,base_filter)
        self.body = DNLNonlinearMapping(base_filter,self.nums_msc,self.attentiontype,self.nums_DRAblk)
        self.tail = DNLReconstruction(base_filter)
    def forward(self,x):
        out = self.head(x)
        out = self.body(out)
        out = self.tail(out)
        return out





