from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from NPP_LIKE.base_networks import ConvBlock,DeconvBlock

class conv_block(nn.Module):
    def __init__(self,cin,cout,k=3,s=1,p=1,act='relu',bn='batch'):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(cin,cout,kernel_size=k,stride=s,padding=p,activation=act,norm=bn),
            ConvBlock(cout, cout, kernel_size=k, stride=s, padding=p, activation=act, norm=bn)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class conv_cat_block(nn.Module):
    def __init__(self,cin,cout,act='relu',bn='batch'):
        super(conv_cat_block,self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(cin,cout,kernel_size=1,stride=1,padding=0,activation=act,norm=bn),
            ConvBlock(cout, cout, kernel_size=3, stride=1, padding=1, activation=act, norm=bn),
            ConvBlock(cout, cout, kernel_size=3, stride=1, padding=1, activation=act, norm=bn)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,cin,cout,kernel_size=3,stride=1,padding=1,act='relu',bn=None):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            DeconvBlock(cin,cout,kernel_size=kernel_size,stride=stride,padding=padding,activation=act,norm=bn)
        )
    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self,cin,cout,base_filter=64,conv_act='relu',conv_bn='batch',deconv_act='relu',deconv_bn=None,tail_act=None,tail_bn=None):
        super(U_Net,self).__init__()
        n1 = base_filter
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Conv1 = conv_block(cin,filters[0],act=conv_act,bn=conv_bn)
        self.Conv2 = conv_block(filters[0], filters[1], act=conv_act, bn=conv_bn)
        self.Conv3 = conv_block(filters[1], filters[2], act=conv_act, bn=conv_bn)
        self.Conv4 = conv_block(filters[2], filters[3], act=conv_act, bn=conv_bn)
        self.Conv5 = conv_block(filters[3], filters[4], act=conv_act, bn=conv_bn)

        self.Up5 = up_conv(filters[4],filters[3],act=deconv_act,bn=deconv_bn)
        self.Up_conv5 = conv_cat_block(filters[4], filters[3], act=deconv_act, bn=deconv_bn)

        self.Up4 = up_conv(filters[3], filters[2], act=deconv_act, bn=deconv_bn)
        self.Up_conv4 = conv_cat_block(filters[3], filters[2], act=deconv_act, bn=deconv_bn)

        self.Up3 = up_conv(filters[2], filters[1], act=deconv_act, bn=deconv_bn)
        self.Up_conv3 = conv_cat_block(filters[2], filters[1], act=deconv_act, bn=deconv_bn)

        self.Up2 = up_conv(filters[1], filters[0], act=deconv_act, bn=deconv_bn)
        self.Up_conv2 = conv_cat_block(filters[1], filters[0], act=deconv_act, bn=deconv_bn)

        self.Conv = ConvBlock(filters[0],cout,kernel_size=1,stride=1,padding=0,activation=tail_act,norm=tail_bn)

    def forward(self, x):

        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)
        e5 = self.Conv5(e4)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out