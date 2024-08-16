import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# class GradientLoss(nn.Module):
#     def __init__(self):
#         super(GradientLoss, self).__init__()
#         self.filter_x = nn.Conv2d(1, 1, kernel_size=3, padding=1)
#         self.filter_y = nn.Conv2d(1, 1, kernel_size=3, padding=1)
#         # self.loss_fn = torch.nn.L1Loss(reduction = 'sum')
#         self.loss_fn = torch.nn.MSELoss(reduction='sum')
#         self.weights_init_sobel()
#
#     def weights_init_sobel(self):
#         sobel_gx = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
#         sobel_gy = torch.Tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
#         self.filter_x.weight.data.copy_(sobel_gx)
#         self.filter_x.bias.data.fill_(0)
#         self.filter_y.weight.data.copy_(sobel_gy)
#         self.filter_y.bias.data.fill_(0)
#
#     def forward(self, X, Y):
#         input_X = torch.mean(X, 1, True)
#         input_Y = torch.mean(Y, 1, True)
#         x = self.filter_x(input_X)
#         x = torch.abs(x)
#         y = self.filter_y(input_X)
#         y = torch.abs(y)
#         G_X = torch.add(x, y)
#
#         x = self.filter_x(input_Y)
#         x = torch.abs(x)
#         y = self.filter_y(input_Y)
#         y = torch.abs(y)
#         G_Y = torch.add(x, y)
#
#         loss = self.loss_fn(G_X, G_Y)
#         # loss = F.mse_loss(G_X, G_Y, reduction='sum')
#         # loss = F.mse_loss(G_X, G_Y,size_average=True)
#         return loss

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.filter_x = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.filter_y = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        # self.loss_fn = torch.nn.L1Loss(reduction = 'sum')
        # self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.weights_init_sobel()

    def weights_init_sobel(self):
        sobel_gx = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
        sobel_gy = torch.Tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
        self.filter_x.weight.data.copy_(sobel_gx)
        self.filter_x.bias.data.fill_(0)
        self.filter_y.weight.data.copy_(sobel_gy)
        self.filter_y.bias.data.fill_(0)

    def forward(self, X, Y):
        input_X = torch.mean(X, 1, True)
        input_Y = torch.mean(Y, 1, True)
        x = self.filter_x(input_X)
        x = torch.abs(x)
        y = self.filter_y(input_X)
        y = torch.abs(y)
        G_X = torch.add(x, y)

        x = self.filter_x(input_Y)
        x = torch.abs(x)
        y = self.filter_y(input_Y)
        y = torch.abs(y)
        G_Y = torch.add(x, y)

        # loss = self.loss_fn(G_X, G_Y)
        return G_X,G_Y

class GANLoss(torch.nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0,reduction='mean'):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        elif self.gan_type == 'lsgan':
            self.loss = torch.nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

