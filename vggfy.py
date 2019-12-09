import torch.nn as nn
import torch
import numpy as np
from utils.dconv import Dconv_cos, Dconv_euc, Dconv_rand, Dconv_drop, Dconv_shuffle, Dconv_shuffleall, Dconv_none
from utils.dconv import Dconv_cshuffle, Dconv_crand, Dconv_localshuffle


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

class CONV1D_1x1(nn.Module):
    """
    This is just a wraper for a CONV_1x1
    """
    def __init__(self, inplanes, outplanes, bias):
        super(CONV1D_1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV1D_3x3(nn.Module):
    """
    In order to show that spatial relation is not important, I do GAP after 22 layer,
    then I just apply conv1D onto the channels until the very end
    """
    def __init__(self, inplanes, outplanes, bias=False):
        super(CONV1D_3x3, self).__init__()
        self.conv1d = nn.Linear(inplanes, outplanes, bias=bias)
        self.bn = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV_3x3shuffle(nn.Module):
    """
    This is just a wraper for a CONV_3x3shuffle
    """
    def __init__(self, inplanes, outplanes, kernelsize, stride, padding):
        super(CONV_3x3shuffle, self).__init__()
        self.dconv = Dconv_shuffle(inplanes, outplanes, kernelsize, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CONV_3x3rand(nn.Module):
    """
    This is just a wraper for a CONV_3x3rand
    """
    def __init__(self, inplanes, outplanes, kernelsize, stride, padding, bias):
        super(CONV_3x3rand, self).__init__()
        self.dconv = Dconv_rand(inplanes, outplanes, kernelsize, stride, padding, bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dconv(x)
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
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv34 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv44 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        self.conv54 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
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


class VGG16_Shuffle(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        """
        only one layer will be shuffled here
        """
        super(VGG16_Shuffle, self).__init__()
        print("CIFAR VGG16_Shuffle is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        # Define the building blocks
        if layer == 11:
            self.conv11 = CONV_3x3shuffle(3, 64, kernelsize=3, stride=1, padding=1)
        else:
            self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, bias=False)

        if layer == 12:
            self.conv12 = nn.Sequential(CONV_3x3shuffle(64, 64, kernelsize=3, stride=1, padding=1),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv12 = nn.Sequential(CONV_3x3(64, 64, kernelsize=3, stride=1, padding=1, bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding=1, bias=False)

        if layer == 21:
            self.conv21 = CONV_3x3shuffle(64, 128, kernelsize=3, stride=1, padding=1)
        else:
            self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, bias=False)

        if layer == 22:
            self.conv22 = nn.Sequential(CONV_3x3shuffle(128, 128, kernelsize=3, stride=1, padding=1),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv22 = nn.Sequential(CONV_3x3(128, 128, kernelsize=3, stride=1, padding=1, bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding=1, bias=False)

        if layer == 31:
            self.conv31 = CONV_3x3shuffle(128, 256, kernelsize=3, stride=1, padding=1)
        else:
            self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, bias=False)

        if layer == 32:
            self.conv32 = CONV_3x3shuffle(256, 256, kernelsize=3, stride=1, padding=1)
        else:
            self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=False)

        if layer == 33:
            self.conv33 = nn.Sequential(CONV_3x3shuffle(256, 256, kernelsize=3, stride=1, padding=1),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv33 = nn.Sequential(CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding=1, bias=False)

        if layer == 41:
            self.conv41 = CONV_3x3shuffle(256, 512, kernelsize=3, stride=1, padding=1)
        else:
            self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, bias=False)

        if layer == 42:
            self.conv42 = CONV_3x3shuffle(512, 512, kernelsize=3, stride=1, padding=1)
        else:
            self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=False)

        if layer == 43:
            self.conv43 = nn.Sequential(CONV_3x3shuffle(512, 512, kernelsize=3, stride=1, padding=1),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv43 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
            # self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding=1, bias=False)

        if layer == 51:
            self.conv51 = CONV_3x3shuffle(512, 512, kernelsize=3, stride=1, padding=1)
        else:
            self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=False)

        if layer == 52:
            self.conv52 = CONV_3x3shuffle(512, 512, kernelsize=3, stride=1, padding=1)
        else:
            self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=False)

        if layer == 53:
            self.conv53 = nn.Sequential(CONV_3x3shuffle(512, 512, kernelsize=3, stride=1, padding=1),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv53 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=False),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        x = self.conv11(input_x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        return self.fc(x)


class VGG16_Rand(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        """
        multiple layers will be rand
        """
        super(VGG16_Rand, self).__init__()
        print("CIFAR VGG16_Rand is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer
        self.bias = True

        # Define the building blocks
        if layer <= 11:
            self.conv11 = CONV_3x3rand(3, 64, kernelsize=3, stride=1, padding=1, bias=self.bias)
        else:
            self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, bias=self.bias)

        if layer <= 12:
            self.conv12 = nn.Sequential(CONV_3x3rand(64, 64, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv12 = nn.Sequential(CONV_3x3(64, 64, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer <= 21:
            self.conv21 = CONV_3x3rand(64, 128, kernelsize=3, stride=1, padding=1, bias=self.bias)
        else:
            self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, bias=self.bias)

        if layer <= 22:
            self.conv22 = nn.Sequential(CONV_3x3rand(128, 128, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv22 = nn.Sequential(CONV_3x3(128, 128, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer <= 31:
            self.conv31 = CONV_3x3rand(128, 256, kernelsize=3, stride=1, padding=1, bias=self.bias)
        else:
            self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, bias=self.bias)

        if layer <= 32:
            self.conv32 = CONV_3x3rand(256, 256, kernelsize=3, stride=1, padding=1, bias=self.bias)
        else:
            self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=self.bias)

        if layer <= 33:
            self.conv33 = nn.Sequential(CONV_3x3rand(256, 256, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv33 = nn.Sequential(CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer <= 41:
            self.conv41 = CONV_3x3rand(256, 512, kernelsize=3, stride=1, padding=1, bias=self.bias)
        else:
            self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, bias=self.bias)

        if layer <= 42:
            self.conv42 = CONV_3x3rand(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias)
        else:
            self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias)

        if layer <= 43:
            self.conv43 = nn.Sequential(CONV_3x3rand(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv43 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer <= 51:
            self.conv51 = CONV_3x3rand(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias)
        else:
            self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias)

        if layer <= 52:
            self.conv52 = CONV_3x3rand(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias)
        else:
            self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias)

        if layer <= 53:
            self.conv53 = nn.Sequential(CONV_3x3rand(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv53 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=self.bias),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        x = self.conv11(input_x)
        x = self.conv12(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        return self.fc(x)


class VGG16_1d(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer, type='none'):
        """
        :param type: this is only for shuffle experiments, remember to get rid of it afterwards!
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        """
        super(VGG16_1d, self).__init__()
        print("CIFAR VGG16_1d is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        # Define the building blocks
        if layer > 11:
            self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv11 = CONV1D_3x3(3, 64, bias=True)

        if layer > 12:
            self.conv12 = nn.Sequential(CONV_3x3(64, 64, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv12 = CONV1D_3x3(64, 64, bias=True)

        if layer > 21:
            self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv21 = CONV1D_3x3(64, 128, bias=True)

        if layer > 22:
            self.conv22 = nn.Sequential(CONV_3x3(128, 128, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv22 = CONV1D_3x3(128, 128, bias=True)

        if layer > 31:
            self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv31 = CONV1D_3x3(128, 256, bias=True)

        if layer > 32:
            self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv32 = CONV1D_3x3(256, 256, bias=True)

        if layer > 33:
            self.conv33 = nn.Sequential(CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv33 = CONV1D_3x3(256, 256, bias=True)

        if layer > 41:
            self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv41 = CONV1D_3x3(256, 512, bias=True)

        if layer > 42:
            self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv42 = CONV1D_3x3(512, 512, bias=True)

        if layer > 43:
            self.conv43 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv43 = CONV1D_3x3(512, 512, bias=True)

        if layer > 51:
            self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv51 = CONV1D_3x3(512, 512, bias=True)

        if layer > 52:
            self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv52 = CONV1D_3x3(512, 512, bias=True)

        if layer > 53:
            self.conv53 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv53 = CONV1D_3x3(512, 512, bias=True)

        # self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # raise Exception('You are using a model without BN!!!')

    def forward(self, x):
        # print('input size:', input_x.size())
        if self.layer == 11:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv11(x)
        if self.layer == 12:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv12(x)
        if self.layer == 21:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv21(x)
        if self.layer == 22:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv22(x)
        if self.layer == 31:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv31(x)
        if self.layer == 32:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv32(x)
        if self.layer == 33:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv33(x)
        if self.layer == 41:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv41(x)
        if self.layer == 42:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv42(x)
        if self.layer == 43:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv43(x)
        if self.layer == 51:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv51(x)
        if self.layer == 52:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv52(x)
        if self.layer == 53:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.conv53(x)
        if self.layer == 99:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/weight52.npy', self.fc.weight.detach().cpu().numpy())

        return self.fc(x)


class VGG16_1x1LMP(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        """
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        """
        super(VGG16_1x1LMP, self).__init__()
        print("VGG16_1x1LMP is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        # Define the building blocks
        if layer > 11:
            self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv11 = CONV1D_1x1(3, 64, bias=True)

        if layer > 12:
            self.conv12 = nn.Sequential(CONV_3x3(64, 64, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv12 = nn.Sequential(CONV1D_1x1(64, 64, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer > 21:
            self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv21 = CONV1D_1x1(64, 128, bias=True)

        if layer > 22:
            self.conv22 = nn.Sequential(CONV_3x3(128, 128, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv22 = nn.Sequential(CONV1D_1x1(128, 128, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer > 31:
            self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv31 = CONV1D_1x1(128, 256, bias=True)

        if layer > 32:
            self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv32 = CONV1D_1x1(256, 256, bias=True)

        if layer > 33:
            self.conv33 = nn.Sequential(CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv33 = nn.Sequential(CONV1D_1x1(256, 256, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer > 41:
            self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv41 = CONV1D_1x1(256, 512, bias=True)

        if layer > 42:
            self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv42 = CONV1D_1x1(512, 512, bias=True)

        if layer > 43:
            self.conv43 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv43 = nn.Sequential(CONV1D_1x1(512, 512, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        if layer > 51:
            self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv51 = CONV1D_1x1(512, 512, bias=True)

        if layer > 52:
            self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv52 = CONV1D_1x1(512, 512, bias=True)

        if layer > 53:
            self.conv53 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv53 = nn.Sequential(CONV1D_1x1(512, 512, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        # self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # raise Exception('You are using a model without BN!!!')

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class VGG16_1x1LAP(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer):
        """
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        """
        super(VGG16_1x1LAP, self).__init__()
        print("VGG16_1x1LAP is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        # Define the building blocks
        if layer > 11:
            self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv11 = CONV1D_1x1(3, 64, bias=True)

        if layer > 12:
            self.conv12 = nn.Sequential(CONV_3x3(64, 64, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv12 = nn.Sequential(CONV1D_1x1(64, 64, bias=True),
                                        nn.AvgPool2d(kernel_size=2, stride=2))

        if layer > 21:
            self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv21 = CONV1D_1x1(64, 128, bias=True)

        if layer > 22:
            self.conv22 = nn.Sequential(CONV_3x3(128, 128, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv22 = nn.Sequential(CONV1D_1x1(128, 128, bias=True),
                                        nn.AvgPool2d(kernel_size=2, stride=2))

        if layer > 31:
            self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv31 = CONV1D_1x1(128, 256, bias=True)

        if layer > 32:
            self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv32 = CONV1D_1x1(256, 256, bias=True)

        if layer > 33:
            self.conv33 = nn.Sequential(CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv33 = nn.Sequential(CONV1D_1x1(256, 256, bias=True),
                                        nn.AvgPool2d(kernel_size=2, stride=2))

        if layer > 41:
            self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv41 = CONV1D_1x1(256, 512, bias=True)

        if layer > 42:
            self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv42 = CONV1D_1x1(512, 512, bias=True)

        if layer > 43:
            self.conv43 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv43 = nn.Sequential(CONV1D_1x1(512, 512, bias=True),
                                        nn.AvgPool2d(kernel_size=2, stride=2))

        if layer > 51:
            self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv51 = CONV1D_1x1(512, 512, bias=True)

        if layer > 52:
            self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True)
        else:
            self.conv52 = CONV1D_1x1(512, 512, bias=True)

        if layer > 53:
            self.conv53 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.conv53 = nn.Sequential(CONV1D_1x1(512, 512, bias=True),
                                        nn.AvgPool2d(kernel_size=2, stride=2))

        # self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Linear(4096, num_classes))

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # raise Exception('You are using a model without BN!!!')

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


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


def vgg16_1d(**kwargs):
    """
    Constructs a VGG16_1d model.
    """
    return VGG16_1d(**kwargs)
