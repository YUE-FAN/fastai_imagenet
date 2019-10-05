import torch.nn as nn
import math
# import torch
# import numpy as np
# from .dconv import DConv1Dai_share, DConv1Dai, Dconv_cos, Dconv_euc, Dconv_rand, Dconv_drop, Dconv_shuffle, Dconv_shuffleall, Dconv_none, Dconv_horizontal, Dconv_vertical
# from .dconv import Dconv_cshuffle, Dconv_crand, Dconv_localshuffle


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual1x1(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual1x1, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 1, stride, 0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, oup, 1, stride, 0, bias=False),
                nn.BatchNorm2d(oup),
                # nn.ReLU6(inplace=True),
                # pw-linear
                # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        print("MobileNetV2 is used")
        block = InvertedResidual
        layers = [conv_3x3_bn(3, 32, 2)]

        layers.append(block(32, 16, 1, 1))

        layers.append(block(16, 24, 2, 6))
        layers.append(block(24, 24, 1, 6))

        layers.append(block(24, 32, 2, 6))
        layers.append(block(32, 32, 1, 6))
        layers.append(block(32, 32, 1, 6))

        layers.append(block(32, 64, 2, 6))
        layers.append(block(64, 64, 1, 6))
        layers.append(block(64, 64, 1, 6))
        layers.append(block(64, 64, 1, 6))

        layers.append(block(64, 96, 1, 6))
        layers.append(block(96, 96, 1, 6))
        layers.append(block(96, 96, 1, 6))

        layers.append(block(96, 160, 2, 6))
        layers.append(block(160, 160, 1, 6))
        layers.append(block(160, 160, 1, 6))

        layers.append(block(160, 320, 1, 6))

        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(320, 1280)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2_1x1(nn.Module):
    def __init__(self, num_classes, layer):
        super(MobileNetV2_1x1, self).__init__()
        print("MobileNetV2_1x1 is used")
        block = InvertedResidual
        block1x1 = InvertedResidual1x1

        assert layer in [10, 20, 21, 30, 31, 32, 40, 41, 42, 43, 50, 51, 52, 60, 61, 62, 70, 99]
        layers = [conv_3x3_bn(3, 32, 2)]

        if layer > 10:
            layers.append(block(32, 16, 1, 1))
        else:
            layers.append(block1x1(32, 16, 1, 1))

        if layer > 20:
            layers.append(block(16, 24, 2, 6))
        else:
            layers.append(block1x1(16, 24, 2, 6))
        if layer > 21:
            layers.append(block(24, 24, 1, 6))
        else:
            layers.append(block1x1(24, 24, 1, 6))

        if layer > 30:
            layers.append(block(24, 32, 2, 6))
        else:
            layers.append(block1x1(24, 32, 2, 6))
        if layer > 31:
            layers.append(block(32, 32, 1, 6))
        else:
            layers.append(block1x1(32, 32, 1, 6))
        if layer > 32:
            layers.append(block(32, 32, 1, 6))
        else:
            layers.append(block1x1(32, 32, 1, 6))

        if layer > 40:
            layers.append(block(32, 64, 2, 6))
        else:
            layers.append(block1x1(32, 64, 2, 6))
        if layer > 41:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))
        if layer > 42:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))
        if layer > 43:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))

        if layer > 50:
            layers.append(block(64, 96, 1, 6))
        else:
            layers.append(block1x1(64, 96, 1, 6))
        if layer > 51:
            layers.append(block(96, 96, 1, 6))
        else:
            layers.append(block1x1(96, 96, 1, 6))
        if layer > 52:
            layers.append(block(96, 96, 1, 6))
        else:
            layers.append(block1x1(96, 96, 1, 6))

        if layer > 60:
            layers.append(block(96, 160, 2, 6))
        else:
            layers.append(block1x1(96, 160, 2, 6))
        if layer > 61:
            layers.append(block(160, 160, 1, 6))
        else:
            layers.append(block1x1(160, 160, 1, 6))
        if layer > 62:
            layers.append(block(160, 160, 1, 6))
        else:
            layers.append(block1x1(160, 160, 1, 6))

        if layer > 70:
            layers.append(block(160, 320, 1, 6))
        else:
            layers.append(block1x1(160, 320, 1, 6))

        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(320, 1280)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2_1x1LMP(nn.Module):
    def __init__(self, num_classes, layer):
        super(MobileNetV2_1x1LMP, self).__init__()
        print("MobileNetV2_1x1LMP is used")
        block = InvertedResidual
        block1x1 = InvertedResidual1x1

        assert layer in [10, 20, 21, 30, 31, 32, 40, 41, 42, 43, 50, 51, 52, 60, 61, 62, 70, 99]
        layers = [conv_3x3_bn(3, 32, 2)]

        if layer > 10:
            layers.append(block(32, 16, 1, 1))
        else:
            layers.append(block1x1(32, 16, 1, 1))

        if layer > 20:
            layers.append(block(16, 24, 2, 6))
        else:
            layers.append(nn.Sequential(block1x1(16, 24, 1, 6), nn.MaxPool2d(2, 2)))
        if layer > 21:
            layers.append(block(24, 24, 1, 6))
        else:
            layers.append(block1x1(24, 24, 1, 6))

        if layer > 30:
            layers.append(block(24, 32, 2, 6))
        else:
            layers.append(nn.Sequential(block1x1(24, 32, 1, 6), nn.MaxPool2d(2, 2)))
        if layer > 31:
            layers.append(block(32, 32, 1, 6))
        else:
            layers.append(block1x1(32, 32, 1, 6))
        if layer > 32:
            layers.append(block(32, 32, 1, 6))
        else:
            layers.append(block1x1(32, 32, 1, 6))

        if layer > 40:
            layers.append(block(32, 64, 2, 6))
        else:
            layers.append(nn.Sequential(block1x1(32, 64, 1, 6), nn.MaxPool2d(2, 2)))
        if layer > 41:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))
        if layer > 42:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))
        if layer > 43:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))

        if layer > 50:
            layers.append(block(64, 96, 1, 6))
        else:
            layers.append(block1x1(64, 96, 1, 6))
        if layer > 51:
            layers.append(block(96, 96, 1, 6))
        else:
            layers.append(block1x1(96, 96, 1, 6))
        if layer > 52:
            layers.append(block(96, 96, 1, 6))
        else:
            layers.append(block1x1(96, 96, 1, 6))

        if layer > 60:
            layers.append(block(96, 160, 2, 6))
        else:
            layers.append(nn.Sequential(block1x1(96, 160, 1, 6), nn.MaxPool2d(2, 2)))
        if layer > 61:
            layers.append(block(160, 160, 1, 6))
        else:
            layers.append(block1x1(160, 160, 1, 6))
        if layer > 62:
            layers.append(block(160, 160, 1, 6))
        else:
            layers.append(block1x1(160, 160, 1, 6))

        if layer > 70:
            layers.append(block(160, 320, 1, 6))
        else:
            layers.append(block1x1(160, 320, 1, 6))

        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(320, 1280)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2_1x1LAP(nn.Module):
    def __init__(self, num_classes, layer):
        super(MobileNetV2_1x1LAP, self).__init__()
        print("MobileNetV2_1x1LAP is used")
        block = InvertedResidual
        block1x1 = InvertedResidual1x1

        assert layer in [10, 20, 21, 30, 31, 32, 40, 41, 42, 43, 50, 51, 52, 60, 61, 62, 70, 99]
        layers = [conv_3x3_bn(3, 32, 2)]

        if layer > 10:
            layers.append(block(32, 16, 1, 1))
        else:
            layers.append(block1x1(32, 16, 1, 1))

        if layer > 20:
            layers.append(block(16, 24, 2, 6))
        else:
            layers.append(nn.Sequential(block1x1(16, 24, 1, 6), nn.AvgPool2d(2, 2)))
        if layer > 21:
            layers.append(block(24, 24, 1, 6))
        else:
            layers.append(block1x1(24, 24, 1, 6))

        if layer > 30:
            layers.append(block(24, 32, 2, 6))
        else:
            layers.append(nn.Sequential(block1x1(24, 32, 1, 6), nn.AvgPool2d(2, 2)))
        if layer > 31:
            layers.append(block(32, 32, 1, 6))
        else:
            layers.append(block1x1(32, 32, 1, 6))
        if layer > 32:
            layers.append(block(32, 32, 1, 6))
        else:
            layers.append(block1x1(32, 32, 1, 6))

        if layer > 40:
            layers.append(block(32, 64, 2, 6))
        else:
            layers.append(nn.Sequential(block1x1(32, 64, 1, 6), nn.AvgPool2d(2, 2)))
        if layer > 41:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))
        if layer > 42:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))
        if layer > 43:
            layers.append(block(64, 64, 1, 6))
        else:
            layers.append(block1x1(64, 64, 1, 6))

        if layer > 50:
            layers.append(block(64, 96, 1, 6))
        else:
            layers.append(block1x1(64, 96, 1, 6))
        if layer > 51:
            layers.append(block(96, 96, 1, 6))
        else:
            layers.append(block1x1(96, 96, 1, 6))
        if layer > 52:
            layers.append(block(96, 96, 1, 6))
        else:
            layers.append(block1x1(96, 96, 1, 6))

        if layer > 60:
            layers.append(block(96, 160, 2, 6))
        else:
            layers.append(nn.Sequential(block1x1(96, 160, 1, 6), nn.AvgPool2d(2, 2)))
        if layer > 61:
            layers.append(block(160, 160, 1, 6))
        else:
            layers.append(block1x1(160, 160, 1, 6))
        if layer > 62:
            layers.append(block(160, 160, 1, 6))
        else:
            layers.append(block1x1(160, 160, 1, 6))

        if layer > 70:
            layers.append(block(160, 320, 1, 6))
        else:
            layers.append(block1x1(160, 320, 1, 6))

        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(320, 1280)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(**kwargs):
    """
    Constructs a MobileNetV2 model
    """
    return MobileNetV2(**kwargs)


def mobilenetv2_1x1(**kwargs):
    """
    Constructs a MobileNetV2_1x1 model
    """
    return MobileNetV2_1x1(**kwargs)


def mobilenetv2_1x1lmp(**kwargs):
    """
    Constructs a MobileNetV1_1x1LMP model.
    """
    return MobileNetV2_1x1LMP(**kwargs)


def mobilenetv2_1x1lap(**kwargs):
    """
    Constructs a MobileNetV1_1x1LAP model.
    """
    return MobileNetV2_1x1LAP(**kwargs)
