import torch.nn as nn
import torch
import numpy as np
# from .dconv import DConv1Dai_share, DConv1Dai, Dconv_cos, Dconv_euc, Dconv_rand, Dconv_drop, Dconv_shuffle, Dconv_shuffleall, Dconv_none, Dconv_horizontal, Dconv_vertical
# from .dconv import Dconv_cshuffle, Dconv_crand


class CONV_3x3(nn.Module):
    """
    This is just a wraper for a conv3x3
    """
    def __init__(self, inplanes, outplanes, kernelsize, stride, padding, bias):
        super(CONV_3x3, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernelsize, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # np.save('/nethome/yuefan/fanyue/dconv/cifar100_weights/weight'+str(n)+'.npy', self.conv.weight.detach().cpu().numpy())
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




def SpatialAttn_whr(x):
    """Spatial Attention"""
    x_shape = x.size()
    a = x.sum(1, keepdim=True)
    a = a.view(x_shape[0], -1)
    a = a / a.sum(1, keepdim=True)
    a = a.view(x_shape[0], 1, x_shape[2], x_shape[3])
    return a


def ChannelAttn_whr(x):
    """Channel Attention"""
    x_shape = x.size()
    x = x.view(x_shape[0], x_shape[1], -1)  # [bs, c, h*w]
    a = x.sum(-1, keepdim=False)  # [bs, c]
    a = a / a.sum(1, keepdim=True)  # [bs, c]
    a = a.unsqueeze(-1).unsqueeze(-1)
    return a


class VGG19(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top):
        super(VGG19, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=1, padding='same', bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=1, padding='same', bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv34 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv44 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=True)
        self.conv54 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AvgPool2d(7)  # TODO: check the final size
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print('input size:', input_x.size())
        x = self.conv11(input_x)
        x = self.conv12(x)
        x = self.pool1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.pool2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv34(x)
        x = self.pool3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv44(x)
        x = self.pool4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.conv54(x)
        x = self.pool5(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class VGG16(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top):

        super(VGG16, self).__init__()
        print("CIFAR VGG16 is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AvgPool2d(7)  # TODO: check the final size
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print('input size:', input_x.size())
        x = self.conv11(input_x)
        x = self.conv12(x)
        x = self.pool1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.pool2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.pool3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.pool4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.pool5(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/fcweight.npy', self.fc.weight.detach().cpu().numpy())
        return self.fc(x)


def vgg19(**kwargs):
    """
    Constructs a VGG19 model.
    """
    return VGG19(**kwargs)


def vgg16(**kwargs):
    """
    Constructs a VGG16 model.
    """
    return VGG16(**kwargs)