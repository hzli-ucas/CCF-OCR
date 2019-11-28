import torch.nn as nn


class Bottleneck(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(nIn, nHidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nHidden)
        self.conv2 = nn.Conv2d(nHidden, nHidden, kernel_size=(1,5),
                               padding=(0,2), bias=False)
        self.bn2 = nn.BatchNorm2d(nHidden)
        self.conv3 = nn.Conv2d(nHidden, nOut, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nOut)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class chsNet(nn.Module):

    def __init__(self, nc, nclass, leakyRelu=False):
        super(chsNet, self).__init__()

        ks = [3, 3, 3, 2, 3, 3, 1]
        ps = [1, 1, 1, 0, 0, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, nclass]

        cnn = nn.Sequential()

        def convOnly(i):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # input: 1x22x220
        cnn.add_module('drop0', nn.Dropout(0.1))
        convRelu(0)
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x11x110
        cnn.add_module('drop1', nn.Dropout(0.2))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 256x5x55
        cnn.add_module('drop2', nn.Dropout(0.2))
        convRelu(4)
        convRelu(5)  # 512x1x50

        cnn.add_module('res0', Bottleneck(512, 128, 512))
        cnn.add_module('res1', Bottleneck(512, 128, 512))
        cnn.add_module('drop3', nn.Dropout(0.5))
        convOnly(6)  # nclass x1x50
        self.cnn = cnn

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        return conv


class digitsNet(nn.Module):

    def __init__(self, nc, nclass, leakyRelu=False):
        super(digitsNet, self).__init__()

        ks = [3, 3, 3, 2, 3, 3, 1]
        ps = [(1,9), 1, 1, 0, 0, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, nclass]

        cnn = nn.Sequential()

        def convOnly(i):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # input: 1x22x220
        cnn.add_module('drop0', nn.Dropout(0.1))
        convRelu(0)
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x11x110
        cnn.add_module('drop1', nn.Dropout(0.2))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 256x5x55
        cnn.add_module('drop1', nn.Dropout(0.2))
        convRelu(4)
        convRelu(5)  # 512x1x50
        cnn.add_module('drop3', nn.Dropout(0.5))
        convOnly(6)  # nclass x1x50

        self.cnn = cnn

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        return conv