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
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv34 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv44 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv54 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

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

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv34(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv44(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.conv54(x)
        # print("feature shape:", x.size())

        if self.include_top:
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
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding=1, bias=False)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding=1, bias=False)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding=1, bias=False)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding=1, bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding=1, bias=False)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding=1, bias=False)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding=1, bias=False)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding=1, bias=False)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=False)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding=1, bias=False)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=False)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding=1, bias=False)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding=1, bias=False)

        # self.dropout = nn.Dropout(p=0.5)
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
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/fcweight.npy', self.fc.weight.detach().cpu().numpy())
        #
        #     x_shuff = torch.empty(x.size(0), x.size(1)).cuda(0)
        #     perm = torch.randperm(x.size(1))
        #     x_shuff[:, :] = x[:, perm]
        #
        return self.fc(x)


class VGG16_1d(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer, is_shuff, type='none'):
        """
        :param type: this is only for shuffle experiments, remember to get rid of it afterwards!
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        :param is_shuff: boolean, whether using CONV1D_3x3 or CONV1Dshuff_3x3
        """
        super(VGG16_1d, self).__init__()
        print("CIFAR VGG16_1d is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        if is_shuff:
            conv_block = CONV1Dshuff_3x3
            print('shuff')
        else:
            conv_block= CONV1D_3x3

        # Define the building blocks
        if layer > 11:
            self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv11 = conv_block(3, 64, bias=False)

        if layer > 12:
            self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv12 = conv_block(64, 64, bias=False)

        if layer > 21:
            self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv21 = conv_block(64, 128, bias=False)

        if layer > 22:
            self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv22 = conv_block(128, 128, bias=False)

        if layer > 31:
            self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv31 = conv_block(128, 256, bias=False)

        if layer > 32:
            self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv32 = conv_block(256, 256, bias=False)

        if layer > 33:
            self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv33 = conv_block(256, 256, bias=False)

        if layer > 41:
            self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv41 = conv_block(256, 512, bias=False)

        if layer > 42:
            self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv42 = conv_block(512, 512, bias=False)

        if layer > 43:
            self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv43 = conv_block(512, 512, bias=False)

        if layer > 51:
            self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv51 = conv_block(512, 512, bias=False)

        if layer > 52:
            self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        else:
            self.conv52 = conv_block(512, 512, bias=False)

        if layer > 53:
            self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)
        else:
            self.conv53 = conv_block(512, 512, bias=False)

        # self.dropout = nn.Dropout(p=0.5)
        if layer == 11 or layer == 12:
            s = 32
        elif layer == 21 or layer == 22:
            s = 16
        elif layer == 31 or layer == 32 or layer == 33:
            s = 8
        elif layer == 41 or layer == 42 or layer == 43:
            s = 4
        elif layer == 51 or layer == 52 or layer == 53:
            s = 2
        self.avgpool = nn.AvgPool2d(s)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                raise Exception('You are using a model without BN!!!')
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                raise Exception('You are using a model without BN!!!')

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
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/weight52.npy', self.fc.weight.detach().cpu().numpy())

        return self.fc(x)


class VGG16_SA(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top, layer, type='none'):
        """
        :param type: this is only for shuffle experiments, remember to get rid of it afterwards!
        :param layer: int, if the conv number is smaller than the layer, normal conv is used; otherwise dconv
        """
        super(VGG16_SA, self).__init__()
        print("CIFAR VGG16_SA is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('input size:', input_x.size())
        if self.layer == 11:
            x = x * SpatialAttn_whr(x)
        x = self.conv11(x)
        if self.layer == 12:
            x = x * SpatialAttn_whr(x)
        x = self.conv12(x)
        if self.layer == 21:
            x = x * SpatialAttn_whr(x)
        x = self.conv21(x)
        if self.layer == 22:
            x = x * SpatialAttn_whr(x)
        x = self.conv22(x)
        if self.layer == 31:
            x = x * SpatialAttn_whr(x)
        x = self.conv31(x)
        if self.layer == 32:
            x = x * SpatialAttn_whr(x)
        x = self.conv32(x)
        if self.layer == 33:
            x = x * SpatialAttn_whr(x)
        x = self.conv33(x)
        if self.layer == 41:
            x = x * SpatialAttn_whr(x)
        x = self.conv41(x)
        if self.layer == 42:
            x = x * SpatialAttn_whr(x)
        x = self.conv42(x)
        if self.layer == 43:
            x = x * SpatialAttn_whr(x)
        x = self.conv43(x)
        if self.layer == 51:
            x = x * SpatialAttn_whr(x)
        x = self.conv51(x)
        if self.layer == 52:
            x = x * SpatialAttn_whr(x)
        x = self.conv52(x)
        if self.layer == 53:
            x = x * SpatialAttn_whr(x)
        x = self.conv53(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/fcweight.npy', self.fc.weight.detach().cpu().numpy())
        return self.fc(x)


class VGG16_3d(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top):
        super(VGG16_3d, self).__init__()
        print("CIFAR VGG16_3d is used")
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv21 = CONV3D_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv22 = CONV3D_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv43 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)

        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print('input size:', input_x.size())
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
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
            # np.save('/nethome/yuefan/fanyue/dconv/weight52.npy', self.fc.weight.detach().cpu().numpy())
            x = self.fc(x)
        return x


class VGG16_Transpose(nn.Module):  # TODO: try different config of the channels
    def __init__(self, dropout_rate, num_classes, include_top):
        super(VGG16_Transpose, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
        # self.conv11 = DCONV_3x3(3, 64)
        self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv32 = CONV_3x3(256, 484, kernelsize=3, stride=1, padding='same', bias=False)
        self.conv33 = CONV_3x3(64, 484, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv41 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
        self.conv42 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
        # self.conv42 = DCONV_3x3(512, 512)
        self.conv43 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)

        self.conv51 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
        self.conv52 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
        self.conv53 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)

        # self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AvgPool2d(11)  # TODO: check the final size
        self.fc = nn.Linear(484, num_classes)

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

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = x.view(x.size(0), 22, 22, 64)
        x = x.permute(0, 3, 1, 2)
        x = self.conv33(x)

        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv41(x)
        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv42(x)
        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv43(x)

        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv51(x)
        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv52(x)
        x = x.view(x.size(0), 22, 22, 121)
        x = x.permute(0, 3, 1, 2)
        x = self.conv53(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # x = self.dropout(x)
            # TODO: why there is no dropout
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


def vgg16_3d(**kwargs):
    """
    Constructs a VGG16_3d model.
    """
    return VGG16_3d(**kwargs)


def vgg16_1d(**kwargs):
    """
    Constructs a VGG16_1d model.
    """
    return VGG16_1d(**kwargs)


def vgg16_transpose(**kwargs):
    """
    Constructs a vgg16_transpose model.
    """
    return VGG16_Transpose(**kwargs)


def vgg16_sa(**kwargs):
    """
    Constructs a vgg16_sa model.
    """
    return VGG16_SA(**kwargs)

# class VGG16_Transpose(nn.Module):  # TODO: try different config of the channels
#     def __init__(self, dropout_rate, num_classes, include_top):
#         super(VGG16_Transpose, self).__init__()
#         self.dropout_rate = dropout_rate
#         self.num_classes = num_classes
#         self.include_top = include_top
#
#         # Define the building blocks
#         self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv42 = CONV_3x3(512, 484, kernelsize=3, stride=1, padding='same', bias=False)
#         # self.conv42 = DCONV_3x3(512, 512)
#         self.conv43 = CONV_3x3(16, 512, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv53 = CONV_3x3(512, 512, kernelsize=3, stride=2, padding='same', bias=False)
#
#         # self.dropout = nn.Dropout(p=0.5)
#         self.avgpool = nn.AvgPool2d(6)  # TODO: check the final size
#         self.fc = nn.Linear(512, num_classes)
#
#         # Initialize the weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input_x):
#         # print('input size:', input_x.size())
#         x = self.conv11(input_x)
#         x = self.conv12(x)
#
#         x = self.conv21(x)
#         x = self.conv22(x)
#
#         x = self.conv31(x)
#         x = self.conv32(x)
#         x = self.conv33(x)
#
#         x = self.conv41(x)
#
#         x = self.conv42(x)  # 128, 484, 4, 4
#         x = x.view(x.size(0), 22, 22, 16)
#         x = x.permute(0, 3, 1, 2)  # 128, 16, 22, 22
#         x = self.conv43(x)  # 128, 512, 11, 11
#
#         x = self.conv51(x)
#         x = self.conv52(x)
#         x = self.conv53(x)
#         # print("feature shape:", x.size())
#
#         if self.include_top:
#             x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#             # x = self.dropout(x)
#             # TODO: why there is no dropout
#             x = self.fc(x)
#         return x


# class VGG16_Transpose(nn.Module):  # TODO: try different config of the channels
#     # This one is the first one that achieves 52.44%!!!!
#     def __init__(self, dropout_rate, num_classes, include_top):
#         super(VGG16_Transpose, self).__init__()
#         self.dropout_rate = dropout_rate
#         self.num_classes = num_classes
#         self.include_top = include_top
#
#         # Define the building blocks
#         self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv12 = CONV_3x3(64, 64, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv22 = CONV_3x3(128, 128, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv33 = CONV_3x3(256, 256, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=False)
#         # self.conv42 = DCONV_3x3(512, 512)
#         self.conv43 = CONV_3x3(512, 484, kernelsize=3, stride=2, padding='same', bias=False)
#
#         self.conv51 = CONV_3x3(4, 484, kernelsize=3, stride=2, padding='same', bias=False)
#         self.conv52 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
#         self.conv53 = CONV_3x3(121, 484, kernelsize=3, stride=2, padding='same', bias=False)
#
#         # self.dropout = nn.Dropout(p=0.5)
#         # self.avgpool = nn.AvgPool2d(6)  # TODO: check the final size
#         self.fc = nn.Linear(484, num_classes)
#
#         # Initialize the weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input_x):
#         # print('input size:', input_x.size())
#         x = self.conv11(input_x)
#         x = self.conv12(x)
#
#         x = self.conv21(x)
#         x = self.conv22(x)
#
#         x = self.conv31(x)
#         x = self.conv32(x)
#         x = self.conv33(x)
#
#         x = self.conv41(x)
#         x = self.conv42(x)
#         x = self.conv43(x)
#
#         x = x.view(x.size(0), 22, 22, 4)
#         x = x.permute(0, 3, 1, 2)  # [128, 4, 22, 22]
#         x = self.conv51(x)  # [128, 484, 11, 11]
#         x = x.view(x.size(0), 22, 22, 121)
#         x = x.permute(0, 3, 1, 2)
#         x = self.conv52(x)
#         x = x.view(x.size(0), 22, 22, 121)
#         x = x.permute(0, 3, 1, 2)  # [128, 121, 22, 22]
#         x = self.conv53(x)
#         x = x.view(x.size(0), 22, 22, 121)
#         x = x.permute(0, 3, 1, 2)  # [128, 121, 22, 22]
#         x = torch.mean(x, dim=1)
#         # print("feature shape:", x.size())
#
#         if self.include_top:
#             # x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#             # x = self.dropout(x)
#             # TODO: why there is no dropout
#             x = self.fc(x)
#         return x