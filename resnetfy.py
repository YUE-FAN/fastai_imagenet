import torch.nn as nn
import torch
import numpy as np
from .dconv import DConv1Dai_share, DConv1Dai, Dconv_cos, Dconv_euc, Dconv_rand, Dconv_drop, Dconv_shuffle, \
    Dconv_shuffleall, Dconv_none, Dconv_horizontal, Dconv_vertical, Dconv_cshuffle, Dconv_crand


class identity_block_1D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super(identity_block_1D, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Linear(plane1, plane2, bias=False)
        # self.bn2 = nn.BatchNorm1d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        # out = self.bn1(out)
        out = self.relu(out)

        out = out.view(out.size(0), -1)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = out.view(out.size(0), out.size(1), 1, 1)
        out = self.conv3(out)
        # out = self.bn3(out)

        out += input_tensor
        out = self.relu(out)
        return out


class bottleneck_1D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck_1D, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Linear(plane1, plane2, bias=False)
        # self.bn2 = nn.BatchNorm1d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(plane3)
        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        # out = self.bn1(out)
        out = self.relu(out)

        out = out.view(out.size(0), -1)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = out.view(out.size(0), out.size(1), 1, 1)
        out = self.conv3(out)
        # out = self.bn3(out)

        shortcut = self.conv4(input_tensor)
        # shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class identity_block_1D_shuff(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super(identity_block_1D_shuff, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Linear(plane1, plane2, bias=False)
        self.bn2 = nn.BatchNorm1d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        x_shape = input_tensor.size()  # [128, 3, 32, 32]
        shuffled_input = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(0)
        perm = torch.randperm(x_shape[1])
        shuffled_input[:, :, :, :] = input_tensor[:, perm, :, :]

        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = out.view(out.size(0), -1)
        x_shuff = torch.empty(out.size(0), out.size(1)).cuda(0)
        perm = torch.randperm(out.size(1))
        x_shuff[:, :] = out[:, perm]
        out = self.conv2(x_shuff)
        out = self.bn2(out)
        out = self.relu(out)

        out = out.view(out.size(0), out.size(1), 1, 1)
        out = self.conv3(out)
        out = self.bn3(out)

        out += shuffled_input
        out = self.relu(out)
        return out


class bottleneck_1D_shuff(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck_1D_shuff, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Linear(plane1, plane2, bias=False)
        self.bn2 = nn.BatchNorm1d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = out.view(out.size(0), -1)
        x_shuff = torch.empty(out.size(0), out.size(1)).cuda(0)
        perm = torch.randperm(out.size(1))
        x_shuff[:, :] = out[:, perm]
        out = self.conv2(x_shuff)
        out = self.bn2(out)
        out = self.relu(out)

        out = out.view(out.size(0), out.size(1), 1, 1)
        out = self.conv3(out)
        out = self.bn3(out)

        x_shape = input_tensor.size()  # [128, 3, 32, 32]
        shuffled_input = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(0)
        perm1 = torch.randperm(x_shape[1])
        shuffled_input[:, :, :, :] = input_tensor[:, perm1, :, :]
        shortcut = self.conv4(shuffled_input)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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


def conv_1_3x3():
    return nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),  # 'SAME'
                         nn.BatchNorm2d(16),
                         nn.ReLU(inplace=True))
                         # TODO: nn.MaxPool2d(kernel_size=3, stride=2, padding=0))  # 'valid'


def conv_1_3x3_dconv():
    return nn.Sequential(Dconv_shuffle(3, 16, 3, 1, 1),
                         # DConv1Dai(3),
                         # nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),  # 'SAME'
                         nn.BatchNorm2d(16),
                         nn.ReLU(inplace=True))
                         # TODO: nn.MaxPool2d(kernel_size=3, stride=2, padding=0))  # 'valid'


def conv_1D():
    return nn.Sequential(nn.Linear(3, 16, bias=False),
                         nn.BatchNorm1d(16),
                         nn.ReLU(inplace=True))
                         # TODO: nn.MaxPool2d(kernel_size=3, stride=2, padding=0))  # 'valid'


class identity_block3(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super(identity_block3, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += input_tensor
        out = self.relu(out)
        return out


class CONV_3x3_skip(nn.Module):
    def __init__(self, inplanes, outplanes, kernelsize, stride, padding, bias):
        super(CONV_3x3_skip, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernelsize, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(outplanes)

        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.maxpool = nn.MaxPool2d(1, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        shape = input_tensor.size()  # none, 484, 8, 8
        out = input_tensor.view(shape[0], np.sqrt(shape[1]), np.sqrt(shape[1]), shape[2] * shape[3])
        out = out.permute(0, 3, 1, 2)  # none, 64, 22, 22

        shortcut = self.maxpool(out)  # none, 64, 11, 11
        out = self.conv1(out)  # none, 484, 11, 11
        out = self.bn1(out)

        shortcut = self.conv2(shortcut)
        shortcut = self.bn2(shortcut)  # none, 484, 11, 11

        out += shortcut
        out = self.relu(out)
        return out  # none, 484, 11, 11


class identity_block3_dconv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, type='error'):
        super(identity_block3_dconv, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(plane1)

        # self.offset = DConv1Dai(plane1)
        # self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        # self.dconv = Dconv_cos(8, 8, plane1, plane2)  # TODO: don't use hard code here
        # if type=='none':
        #     self.dconv = Dconv_none(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        # elif type=='shuffleall':
        #     self.dconv = Dconv_shuffleall(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        # elif type=='shuffle':
        #     self.dconv = Dconv_shuffle(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        # elif type=='cshuffle':
        #     self.dconv = Dconv_cshuffle(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        # elif type=='rand':
        #     self.dconv = Dconv_rand(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        # elif type=='crand':
        #     self.dconv = Dconv_crand(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        # else:
        #     raise Exception('The type of the dconv does not exit')
        self.dconv = Dconv_shuffle(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(plane2)

        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        # out = self.bn1(out)
        out = self.relu(out)

        # out = self.offset(out)
        # out = self.conv2(out)
        out = self.dconv(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        type = 'shuffle'  # TODO:!!!!!!!!!!!!!!

        if type=='none':
            shuffled_input = input_tensor
        elif type=='shuffleall':
            x_shape = input_tensor.size()  # [128, 3, 32, 32]
            x = input_tensor.view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 3*32*32]
            shuffled_input = torch.empty(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]).cuda(0)
            perm = torch.randperm(x_shape[1] * x_shape[2] * x_shape[3])
            shuffled_input[:, :] = x[:, perm]
            shuffled_input = shuffled_input.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        elif type=='shuffle':
            x_shape = input_tensor.size()  # [128, 3, 32, 32]
            x = input_tensor.view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 3*32*32]
            shuffled_input = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(0)
            perm = torch.empty(0).float()
            for i in range(x_shape[1]):
                a = torch.randperm(x_shape[2] * x_shape[3]) + i * x_shape[2] * x_shape[3]
                perm = torch.cat((perm, a.float()), 0)
            shuffled_input[:, :, :, :] = x[:, perm.long()].view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        elif type=='cshuffle':
            x_shape = input_tensor.size()  # [128, 3, 32, 32]
            x = input_tensor.permute(0, 2, 3, 1)  # [128, 32, 32, 3]
            x = x.contiguous().view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 32*32*3]
            shuffled_input = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(0)
            perm = torch.empty(0).float()
            for i in range(x_shape[2] * x_shape[3]):
                a = torch.randperm(x_shape[1]) + i * x_shape[1]
                perm = torch.cat((perm, a.float()), 0)
            shuffled_input[:, :, :, :] = x[:, perm.long()].view(x_shape[0], x_shape[2], x_shape[3], x_shape[1]).permute(0, 3,
                                                                                                                  1, 2)
        elif type=='rand':
            x_shape = input_tensor.size()  # [128, 3, 32, 32]
            x = input_tensor.view(x_shape[0], x_shape[1], x_shape[2] * x_shape[3])  # [128, 3, 32*32]
            shuffled_input = torch.empty(x_shape[0], x_shape[1], x_shape[2] * x_shape[3]).cuda(0)
            perm = torch.randperm(x_shape[2] * x_shape[3])
            shuffled_input[:, :, :] = x[:, :, perm]
            shuffled_input = shuffled_input.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        elif type=='crand':
            x_shape = input_tensor.size()  # [128, 3, 32, 32]
            shuffled_input = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(0)
            perm = torch.randperm(x_shape[1])
            shuffled_input[:, :, :, :] = input_tensor[:, perm, :, :]
        else:
            raise Exception('The type of the dconv does not exit')
        out += shuffled_input
        out = self.relu(out)
        return out


class bottleneck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.conv4(input_tensor)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class bottleneck_tr(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck_tr, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.conv4(input_tensor)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class bottleneck_dconv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2), type='error'):
        super(bottleneck_dconv, self).__init__()
        plane1, plane2, plane3 = planes

        # if type=='none':
        #     self.dconv1 = Dconv_none(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        #     self.dconv2 = Dconv_none(inplanes, plane3, kernel_size=1, stride=strides, padding=0)
        # elif type=='shuffleall':
        #     self.dconv1 = Dconv_shuffleall(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        #     self.dconv2 = Dconv_shuffleall(inplanes, plane3, kernel_size=1, stride=strides, padding=0)
        # elif type=='shuffle':
        #     self.dconv1 = Dconv_shuffle(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        #     self.dconv2 = Dconv_shuffle(inplanes, plane3, kernel_size=1, stride=strides, padding=0)
        # elif type=='cshuffle':
        #     self.dconv1 = Dconv_cshuffle(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        #     self.dconv2 = Dconv_cshuffle(inplanes, plane3, kernel_size=1, stride=strides, padding=0)
        # elif type=='rand':
        #     self.dconv1 = Dconv_rand(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        #     self.dconv2 = Dconv_rand(inplanes, plane3, kernel_size=1, stride=strides, padding=0)
        # elif type=='crand':
        #     self.dconv1 = Dconv_crand(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        #     self.dconv2 = Dconv_crand(inplanes, plane3, kernel_size=1, stride=strides, padding=0)
        # else:
        #     raise Exception('The type of the dconv does not exit')

        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=strides, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(plane1)

        # self.offset = DConv1Dai(plane1)
        # self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        # self.dconv = Dconv_drop(8, 8, plane1, plane2)  # TODO: don't use hard code here
        self.dconv1 = Dconv_shuffle(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(plane2)

        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(plane3)

        # self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=strides, padding=0, bias=False)
        self.dconv2 = Dconv_shuffle(inplanes, plane3, kernel_size=1, stride=strides, padding=0)
        # self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        # out = self.bn1(out)
        out = self.relu(out)

        # out = self.offset(out)
        # out = self.conv2(out)
        out = self.dconv1(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        shortcut = self.dconv2(input_tensor)
        # shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out
# def basic_block(input_tensor, kernel_size, filters, stage, block, training, reuse, strides=(2, 2)):
#     filters1, filters2 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = tf.layers.conv2d(inputs=input_tensor, filters=filters1, kernel_size=kernel_size, strides=strides,
#                          padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '2a')
#     x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name=bn_name_base + '2a')
#     x = tf.nn.relu(x)
#
#     x = tf.layers.conv2d(inputs=x, filters=filters2, kernel_size=kernel_size, strides=1,
#                          padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '2b')
#     x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name=bn_name_base + '2b')
#
#     if strides == (1, 1):
#         shortcut = input_tensor
#     else:
#         shortcut = tf.layers.conv2d(inputs=input_tensor, filters=filters2, kernel_size=1, strides=strides,
#                                     padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '1')
#         shortcut = tf.layers.batch_normalization(shortcut, training=training, reuse=reuse, name=bn_name_base + '1')
#
#     x = x + shortcut
#     x = tf.nn.relu(x)
#     return x
#
#
# def identity_block2(input_tensor, kernel_size, filters, stage, block, training, reuse):
#     filters1, filters2 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = tf.layers.conv2d(inputs=input_tensor, filters=filters1, kernel_size=kernel_size, strides=1,
#                          padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '2a')
#     x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name=bn_name_base + '2a')
#     x = tf.nn.relu(x)
#
#     x = tf.layers.conv2d(inputs=x, filters=filters2, kernel_size=kernel_size, strides=1,
#                          padding='SAME', use_bias=False, reuse=reuse, name=conv_name_base + '2b')
#     x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name=bn_name_base + '2b')
#
#     x = x + input_tensor
#     x = tf.nn.relu(x)
#
#     return x


class bottleneck_save(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck_save, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        np.save('/nethome/yuefan/fanyue/dconv/weight.npy', self.conv2.weight.detach().cpu().numpy())
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.conv4(input_tensor)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class Resnet50(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer, type='none'):
        print('resnet50 is used')
        super(Resnet50, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()  # TODO: check if you are using dconv3x3

        if layer > 10:
            self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        else:
            self.bottleneck_1 = bottleneck_dconv(16, [16, 16, 64], kernel_size=3, strides=(1, 1), type=type)
        if layer > 11:
            self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        else:
            self.identity_block_1_1 = identity_block3_dconv(64, [16, 16, 64], kernel_size=3, type=type)
        if layer > 12:
            self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)
        else:
            self.identity_block_1_2 = identity_block3_dconv(64, [16, 16, 64], kernel_size=3, type=type)

        if layer > 20:
            self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        else:
            self.bottleneck_2 = bottleneck_dconv(64, [32, 32, 128], kernel_size=3, strides=(2, 2), type=type)
        if layer > 21:
            self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        else:
            self.identity_block_2_1 = identity_block3_dconv(128, [32, 32, 128], kernel_size=3, type=type)
        if layer > 22:
            self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        else:
            self.identity_block_2_2 = identity_block3_dconv(128, [32, 32, 128], kernel_size=3, type=type)
        if layer > 23:
            self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)
        else:
            self.identity_block_2_3 = identity_block3_dconv(128, [32, 32, 128], kernel_size=3, type=type)

        if layer > 30:
            self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        else:
            self.bottleneck_3 = bottleneck_dconv(128, [64, 64, 256], kernel_size=3, strides=(2, 2), type=type)
        if layer > 31:
            self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_1 = identity_block3_dconv(256, [64, 64, 256], kernel_size=3, type=type)
        if layer > 32:
            self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_2 = identity_block3_dconv(256, [64, 64, 256], kernel_size=3, type=type)
        if layer > 33:
            self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_3 = identity_block3_dconv(256, [64, 64, 256], kernel_size=3, type=type)
        if layer > 34:
            self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_4 = identity_block3_dconv(256, [64, 64, 256], kernel_size=3, type=type)
        if layer > 35:
            self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_5 = identity_block3_dconv(256, [64, 64, 256], kernel_size=3, type=type)

        if layer > 40:
            self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(2, 2))
        else:
            self.bottleneck_4 = bottleneck_dconv(256, [128, 128, 512], kernel_size=3, strides=(2, 2), type=type)
        if layer > 41:
            self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        else:
            self.identity_block_4_1 = identity_block3_dconv(512, [128, 128, 512], kernel_size=3, type=type)
        if layer > 42:
            self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)
        else:
            self.identity_block_4_2 = identity_block3_dconv(512, [128, 128, 512], kernel_size=3, type=type)

        self.avgpool = nn.AvgPool2d(4)  # TODO: check the final size
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                raise Exception('You are using a model without BN!!!')
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # np.save('/nethome/yuefan/fanyue/dconv/fm3x3.npy', x.detach().cpu().numpy())
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class Resnet50_1d(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer, is_shuff, type='none'):
        print('Resnet50_1d is used')
        super(Resnet50_1d, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.layer = layer

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()  # TODO: check if you are using dconv3x3
        # self.conv_3x3 = conv_1D()

        if is_shuff:
            identity = identity_block_1D_shuff
            bottle = bottleneck_1D_shuff
            print('shuff')
        else:
            identity = identity_block_1D
            bottle = bottleneck_1D

        if layer > 10:
            self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        else:
            self.bottleneck_1 = bottle(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        if layer > 11:
            self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        else:
            self.identity_block_1_1 = identity(64, [16, 16, 64], kernel_size=3)
        if layer > 12:
            self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)
        else:
            self.identity_block_1_2 = identity(64, [16, 16, 64], kernel_size=3)

        if layer > 20:
            self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        else:
            self.bottleneck_2 = bottle(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        if layer > 21:
            self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        else:
            self.identity_block_2_1 = identity(128, [32, 32, 128], kernel_size=3)
        if layer > 22:
            self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        else:
            self.identity_block_2_2 = identity(128, [32, 32, 128], kernel_size=3)
        if layer > 23:
            self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)
        else:
            self.identity_block_2_3 = identity(128, [32, 32, 128], kernel_size=3)

        if layer > 30:
            self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        else:
            self.bottleneck_3 = bottle(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        if layer > 31:
            self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_1 = identity(256, [64, 64, 256], kernel_size=3)
        if layer > 32:
            self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_2 = identity(256, [64, 64, 256], kernel_size=3)
        if layer > 33:
            self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_3 = identity(256, [64, 64, 256], kernel_size=3)
        if layer > 34:
            self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_4 = identity(256, [64, 64, 256], kernel_size=3)
        if layer > 35:
            self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)
        else:
            self.identity_block_3_5 = identity(256, [64, 64, 256], kernel_size=3)

        if layer > 40:
            self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(2, 2))
        else:
            self.bottleneck_4 = bottle(256, [128, 128, 512], kernel_size=3, strides=(2, 2))
        if layer > 41:
            self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        else:
            self.identity_block_4_1 = identity(512, [128, 128, 512], kernel_size=3)
        if layer > 42:
            self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)
        else:
            self.identity_block_4_2 = identity(512, [128, 128, 512], kernel_size=3)

        # self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        # self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        # self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)
        # # self.bottleneck_1 = bottleneck_1D(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        # # self.identity_block_1_1 = identity_block_1D(64, [16, 16, 64], kernel_size=3)
        # # self.identity_block_1_2 = identity_block_1D(64, [16, 16, 64], kernel_size=3)
        #
        # self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        # self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        # self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        # self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)
        # # self.bottleneck_2 = bottleneck_1D(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        # # self.identity_block_2_1 = identity_block_1D(128, [32, 32, 128], kernel_size=3)
        # # self.identity_block_2_2 = identity_block_1D(128, [32, 32, 128], kernel_size=3)
        # # self.identity_block_2_3 = identity_block_1D(128, [32, 32, 128], kernel_size=3)
        #
        # self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        # self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        # self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        # self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        # self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        # self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)
        # # self.bottleneck_3 = bottleneck_1D(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        # # self.identity_block_3_1 = identity_block_1D(256, [64, 64, 256], kernel_size=3)
        # # self.identity_block_3_2 = identity_block_1D(256, [64, 64, 256], kernel_size=3)
        # # self.identity_block_3_3 = identity_block_1D(256, [64, 64, 256], kernel_size=3)
        # # self.identity_block_3_4 = identity_block_1D(256, [64, 64, 256], kernel_size=3)
        # # self.identity_block_3_5 = identity_block_1D(256, [64, 64, 256], kernel_size=3)
        #
        # self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(2, 2))
        # self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        # self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)
        # # self.bottleneck_4 = bottleneck_1D(256, [128, 128, 512], kernel_size=3, strides=(2, 2))
        # # self.identity_block_4_1 = identity_block_1D(512, [128, 128, 512], kernel_size=3)
        # # self.identity_block_4_2 = identity_block_1D(512, [128, 128, 512], kernel_size=3)

        # if layer == 11 or layer == 12 or layer == 12 or layer == 12 or layer == 12 or layer == 12 or layer == 12 or layer == 12 or layer == 12 or
        if layer == 10 or layer == 11 or layer == 12 or layer == 20:
            s = 32
        elif layer == 21 or layer == 22 or layer == 23 or layer == 30:
            s = 16
        elif layer == 31 or layer == 32 or layer == 33 or layer == 34 or layer == 35 or layer == 40:
            s = 8
        elif layer == 41 or layer == 42:
            s = 4

        self.avgpool = nn.AvgPool2d(s)  # TODO: check the final size
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_3x3(x)
        # np.save('/nethome/yuefan/fanyue/dconv/fm3x3.npy', x.detach().cpu().numpy())
        if self.layer == 10:
            x = self.avgpool(x)
        x = self.bottleneck_1(x)
        if self.layer == 11:
            x = self.avgpool(x)
        x = self.identity_block_1_1(x)
        if self.layer == 12:
            x = self.avgpool(x)
        x = self.identity_block_1_2(x)

        if self.layer == 20:
            x = self.avgpool(x)
        x = self.bottleneck_2(x)
        if self.layer == 21:
            x = self.avgpool(x)
        x = self.identity_block_2_1(x)
        if self.layer == 22:
            x = self.avgpool(x)
        x = self.identity_block_2_2(x)
        if self.layer == 23:
            x = self.avgpool(x)
        x = self.identity_block_2_3(x)

        if self.layer == 30:
            x = self.avgpool(x)
        x = self.bottleneck_3(x)
        if self.layer == 31:
            x = self.avgpool(x)
        x = self.identity_block_3_1(x)
        if self.layer == 32:
            x = self.avgpool(x)
        x = self.identity_block_3_2(x)
        if self.layer == 33:
            x = self.avgpool(x)
        x = self.identity_block_3_3(x)
        if self.layer == 34:
            x = self.avgpool(x)
        x = self.identity_block_3_4(x)
        if self.layer == 35:
            x = self.avgpool(x)
        x = self.identity_block_3_5(x)

        if self.layer == 40:
            x = self.avgpool(x)
        x = self.bottleneck_4(x)
        if self.layer == 41:
            x = self.avgpool(x)
        x = self.identity_block_4_1(x)
        if self.layer == 42:
            x = self.avgpool(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class Transpose_Resnet50(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        print('Transpose_Resnet50 is used')
        super(Transpose_Resnet50, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()  # TODO: check if you are using dconv3x3

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 484], kernel_size=3, strides=(1, 1))
        self.identity_block_4_1 = CONV_3x3_skip(64, 484, kernelsize=3, stride=2, padding=1, bias=False)
        self.identity_block_4_2 = CONV_3x3_skip(121, 484, kernelsize=3, stride=2, padding=1, bias=False)

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
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # np.save('/nethome/yuefan/fanyue/dconv/fm3x3.npy', x.detach().cpu().numpy())
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class DconvResnet50(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(DconvResnet50, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck_dconv(256, [128, 128, 512], kernel_size=3, strides=(1, 1))
        self.identity_block_4_1 = identity_block3_dconv(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3_dconv(512, [128, 128, 512], kernel_size=3)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class Resnet50_SA(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(Resnet50_SA, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(1, 1))
        self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = x * SpatialAttn_whr(x)
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())

        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            # y = x[0, :, :, :]
            # y = y.sum(0)
            # z = torch.round(y)
            # print(z)
            # x = x * SpatialAttn_whr(x)
            # y = x[0, :, :, :]
            # y = y.sum(0)
            # z = torch.round(y)
            # print(z)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class Resnet50_CA(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(Resnet50_CA, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(1, 1))
        self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = x * ChannelAttn_whr(x)
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())

        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class Resnet50_CASA(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(Resnet50_CASA, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(1, 1))
        self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        xsa = SpatialAttn_whr(x)
        xca = ChannelAttn_whr(x)
        x = x * (xsa * xca)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x


class ResnetWHR(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        super(ResnetWHR, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.num_features = 512

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(2, 2))
        self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)

        # =======================================top=============================================
        self.se1 = SELayer(64)
        self.se2 = SELayer(128)
        self.se3 = SELayer(256)

        self.local_conv_layer1 = nn.Conv2d(64, self.num_features, kernel_size=1, padding=0, bias=False)
        self.local_conv_layer2 = nn.Conv2d(128, self.num_features, kernel_size=1, padding=0, bias=False)
        self.local_conv_layer3 = nn.Conv2d(256, self.num_features, kernel_size=1, padding=0, bias=False)
        self.instance_layer1 = nn.Linear(self.num_features, self.num_classes)
        self.instance_layer2 = nn.Linear(self.num_features, self.num_classes)
        self.instance_layer3 = nn.Linear(self.num_features, self.num_classes)

        self.instance0 = nn.Linear(self.num_features, self.num_classes)
        self.instance1 = nn.Linear(self.num_features, self.num_classes)
        self.instance2 = nn.Linear(self.num_features, self.num_classes)
        self.instance3 = nn.Linear(self.num_features, self.num_classes)
        self.instance4 = nn.Linear(self.num_features, self.num_classes)

        self.local_conv = nn.Conv2d(512, self.num_features, kernel_size=1, padding=0, bias=False)
        self.local_bn = nn.BatchNorm2d(self.num_features)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x_layer1 = self.identity_block_1_2(x)
        # print(x.size())
        x = self.bottleneck_2(x_layer1)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x_layer2 = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x_layer2)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x_layer3 = self.identity_block_3_5(x)
        # print(x.size())
        x = self.bottleneck_4(x_layer3)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        x_layer1 = self.se1(x_layer1)
        x_layer1 = nn.functional.avg_pool2d(x_layer1, kernel_size=(32, 32), stride=(1, 1))
        x_layer1 = self.local_conv_layer1(x_layer1)
        x_layer1 = x_layer1.contiguous().view(x_layer1.size(0), -1)
        x_layer1 = self.instance_layer1(x_layer1)

        x_layer2 = self.se2(x_layer2)
        x_layer2 = nn.functional.avg_pool2d(x_layer2, kernel_size=(16, 16), stride=(1, 1))
        x_layer2 = self.local_conv_layer2(x_layer2)
        x_layer2 = x_layer2.contiguous().view(x_layer2.size(0), -1)
        x_layer2 = self.instance_layer2(x_layer2)

        x_layer3 = self.se3(x_layer3)
        x_layer3 = nn.functional.avg_pool2d(x_layer3, kernel_size=(8, 8), stride=(1, 1))
        x_layer3 = self.local_conv_layer3(x_layer3)
        x_layer3 = x_layer3.contiguous().view(x_layer3.size(0), -1)
        x_layer3 = self.instance_layer3(x_layer3)

        sx = x.size(2) / 4
        x = nn.functional.avg_pool2d(x, kernel_size=(sx, x.size(3)), stride=(sx, x.size(3)))  # 4x1
        x = self.local_conv(x)
        x = self.local_bn(x)
        x = nn.functional.relu(x)

        x4 = nn.functional.avg_pool2d(x, kernel_size=(4, 1), stride=(1, 1))
        x4 = x4.contiguous().view(x4.size(0), -1)
        c4 = self.instance4(x4)
        x = x.chunk(4, dim=2)
        x0 = x[0].contiguous().view(x[0].size(0), -1)
        x1 = x[1].contiguous().view(x[1].size(0), -1)
        x2 = x[2].contiguous().view(x[2].size(0), -1)
        x3 = x[3].contiguous().view(x[3].size(0), -1)
        c0 = self.instance0(x0)
        c1 = self.instance1(x1)
        c2 = self.instance2(x2)
        c3 = self.instance3(x3)
        return c0, c1, c2, c3, c4, x_layer1, x_layer2, x_layer3

# class Resnet34:
#     def __init__(self, training, dropout_rate, num_classes, reuse, include_top):
#         self.training = training
#         self.dropout_rate = dropout_rate
#         self.num_classes = num_classes
#         self.reuse = reuse
#         self.include_top = include_top
#
#     def conv_1_7x7(self, x, reuse):
#         x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[7, 7], strides=2, padding='SAME',
#                              use_bias=False, name='conv_1_7x7', reuse=reuse)
#         x = tf.layers.batch_normalization(x, training=self.training, name='conv_1_7x7', reuse=reuse)
#         x = tf.nn.relu(x)
#         x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='valid', name='conv_1_7x7_pool')
#         return x
#
#     def forward(self, input_x):
#         x = self.conv_1_7x7(input_x, reuse=self.reuse)
#
#         x = basic_block(x, 3, [64, 64], stage=2, block='a', training=self.training, reuse=self.reuse, strides=(1, 1))
#         x = identity_block2(x, 3, [64, 64], stage=2, block='b', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [64, 64], stage=2, block='c', training=self.training, reuse=self.reuse)
#
#         x = basic_block(x, 3, [128, 128], stage=3, block='a', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [128, 128], stage=3, block='b', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [128, 128], stage=3, block='c', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [128, 128], stage=3, block='d', training=self.training, reuse=self.reuse)
#
#         x = basic_block(x, 3, [256, 256], stage=4, block='a', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='b', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='c', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='d', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='e', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [256, 256], stage=4, block='f', training=self.training, reuse=self.reuse)
#
#         x = basic_block(x, 3, [512, 512], stage=5, block='a', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [512, 512], stage=5, block='b', training=self.training, reuse=self.reuse)
#         x = identity_block2(x, 3, [512, 512], stage=5, block='c', training=self.training, reuse=self.reuse)
#
#         print("feature shape:", x.get_shape())
#
#         if self.include_top:
#             x = tf.reduce_mean(x, [1, 2])
#             x = tf.layers.dropout(x, rate=self.dropout_rate, training=self.training)
#             x = tf.layers.dense(inputs=x, use_bias=True, units=self.num_classes, name='linear', reuse=None)
#         return x
def resnet50(**kwargs):
    """
    Constructs a ResNet50 model.
    """
    return Resnet50(**kwargs)


def dconv_resnet50(**kwargs):
    """
    Constructs a DconvResnet50 model.
    """
    return DconvResnet50(**kwargs)


def sa_resnet50(**kwargs):
    """
    Constructs a Resnet50_SA model.
    """
    return Resnet50_SA(**kwargs)


def ca_resnet50(**kwargs):
    """
    Constructs a Resnet50_CA model.
    """
    return Resnet50_CA(**kwargs)


def casa_resnet50(**kwargs):
    """
    Constructs a Resnet50_SACA model.
    """
    return Resnet50_CASA(**kwargs)


def tr_resnet50(**kwargs):
    """
    Constructs a Transpose_Resnet50 model.
    """
    return Transpose_Resnet50(**kwargs)


def d1_resnet50(**kwargs):
    """
    Constructs a Resnet50_1d model.
    """
    return Resnet50_1d(**kwargs)


def resnetwhr(**kwargs):
    """
    Constructs a ResnetWHR model.
    """
    return ResnetWHR(**kwargs)
