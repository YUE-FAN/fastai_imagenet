import torch.nn as nn
import torch


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class Fire_1x1(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire_1x1, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes+expand3x3_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand1x1_activation(self.expand1x1(x))


class SqueezeNet_1x1LMP(nn.Module):
    def __init__(self, num_classes, layer):
        print('SqueezeNet_1x1LMP is used')
        super(SqueezeNet_1x1LMP, self).__init__()
        self.num_classes = num_classes

        possible_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if layer not in possible_layers:
            raise Exception('the layer you choose for SqueezeNet_1x1LMP is invaild!!!')

        self.conv7x7 = nn.Sequential(
                       nn.Conv2d(3, 96, kernel_size=7, stride=2),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
                       )
        if layer > 1:
            self.fire2 = Fire(96, 16, 64, 64)
        else:
            self.fire2 = Fire_1x1(96, 16, 64, 64)
        if layer > 2:
            self.fire3 = Fire(128, 16, 64, 64)
        else:
            self.fire3 = Fire_1x1(128, 16, 64, 64)
        if layer > 3:
            self.fire4 = Fire(128, 32, 128, 128)
        else:
            self.fire4 = Fire_1x1(128, 32, 128, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        if layer > 4:
            self.fire5 = Fire(256, 32, 128, 128)
        else:
            self.fire5 = Fire_1x1(256, 32, 128, 128)
        if layer > 5:
            self.fire6 = Fire(256, 48, 192, 192)
        else:
            self.fire6 = Fire_1x1(256, 150, 192, 192)
        if layer > 6:
            self.fire7 = Fire(384, 48, 192, 192)
        else:
            self.fire7 = Fire_1x1(384, 130, 192, 192)
        if layer > 7:
            self.fire8 = Fire(384, 64, 256, 256)
        else:
            self.fire8 = Fire_1x1(384, 200, 256, 256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        if layer > 8:
            self.fire9 = Fire(512, 64, 256, 256)
        else:
            self.fire9 = Fire_1x1(512, 180, 256, 256)

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv7x7(x)

        x = self.fire2(x)
        x = x + self.fire3(x)
        x = self.fire4(x)
        x = self.pool1(x)

        x = x + self.fire5(x)
        x = self.fire6(x)
        x = x + self.fire7(x)
        x = self.fire8(x)
        x = self.pool2(x)

        x = x + self.fire9(x)

        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


class SqueezeNet_1x1LAP(nn.Module):
    def __init__(self, num_classes, layer):
        print('SqueezeNet_1x1LAP is used')
        super(SqueezeNet_1x1LAP, self).__init__()
        self.num_classes = num_classes

        possible_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if layer not in possible_layers:
            raise Exception('the layer you choose for SqueezeNet_1x1LAP is invaild!!!')

        self.conv7x7 = nn.Sequential(
                       nn.Conv2d(3, 96, kernel_size=7, stride=2),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
                       )
        if layer > 1:
            self.fire2 = Fire(96, 16, 64, 64)
        else:
            self.fire2 = Fire_1x1(96, 16, 64, 64)
        if layer > 2:
            self.fire3 = Fire(128, 16, 64, 64)
        else:
            self.fire3 = Fire_1x1(128, 16, 64, 64)
        if layer > 3:
            self.fire4 = Fire(128, 32, 128, 128)
        else:
            self.fire4 = Fire_1x1(128, 32, 128, 128)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)
        if layer > 4:
            self.fire5 = Fire(256, 32, 128, 128)
        else:
            self.fire5 = Fire_1x1(256, 32, 128, 128)
        if layer > 5:
            self.fire6 = Fire(256, 48, 192, 192)
        else:
            self.fire6 = Fire_1x1(256, 150, 192, 192)
        if layer > 6:
            self.fire7 = Fire(384, 48, 192, 192)
        else:
            self.fire7 = Fire_1x1(384, 130, 192, 192)
        if layer > 7:
            self.fire8 = Fire(384, 64, 256, 256)
        else:
            self.fire8 = Fire_1x1(384, 200, 256, 256)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)
        if layer > 8:
            self.fire9 = Fire(512, 64, 256, 256)
        else:
            self.fire9 = Fire_1x1(512, 180, 256, 256)

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv7x7(x)

        x = self.fire2(x)
        x = x + self.fire3(x)
        x = self.fire4(x)
        x = self.pool1(x)

        x = x + self.fire5(x)
        x = self.fire6(x)
        x = x + self.fire7(x)
        x = self.fire8(x)
        x = self.pool2(x)

        x = x + self.fire9(x)

        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)
