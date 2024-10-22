import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchinfo import summary
import math


class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        # print('------------------------------------------------------------')
        # print("打印x的形状",x.shape)
        # print("打印x挤压的形状", x.squeeze(-1).shape)
        # print("打印x挤压转动的形状", x.squeeze(-1).transpose(-1, -2).shape)
        # print("打印x经过卷积的形状", self.conv1(x.squeeze(-1).transpose(-1, -2)).shape)
        # print("打印x经过卷积转置的形状", self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).shape)
        # print("打印x最终", self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).shape)
        # print('------------------------------------------------------------')
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out


class BasicBlock(nn.Module):      # 左侧的 residual block 结构（18-layer、34-layer）
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):      # 两层卷积 Conv2d + Shutcuts
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.channel = EfficientChannelAttention(planes)       # Efficient Channel Attention module

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        ECA_out = self.channel(out)
        out = out * ECA_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):      # 右侧的 residual block 结构（50-layer、101-layer、152-layer）
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):      # 三层卷积 Conv2d + Shutcuts
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.channel = EfficientChannelAttention(self.expansion*planes)       # Efficient Channel Attention module

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        ECA_out = self.channel(out)
        out = out * ECA_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ECA_ResNet(nn.Module):
    def __init__(self, block, num_blocks,num_classes=1000):
        super(ECA_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)                  # conv1
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)       # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)      # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)      # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)      # conv5_x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.linear(x)
        print("out",out.shape)
        return out


def ECA_ResNet18(num_classes):
    return ECA_ResNet( BasicBlock, [2, 2, 2, 2],num_classes)


def ECA_ResNet34(num_classes):
    return ECA_ResNet(BasicBlock, [3, 4, 6, 3],num_classes)


def ECA_ResNet50(num_classes):
    return ECA_ResNet(Bottleneck, [3, 4, 6, 3],num_classes)


def ECA_ResNet101(num_classes):
    return ECA_ResNet(Bottleneck, [3, 4, 23, 3],num_classes)


def ECA_ResNet152(num_classes):
    return ECA_ResNet(Bottleneck, [3, 8, 36, 3],num_classes)


def test():
    net = ECA_ResNet18(num_classes=6)
    y = net(torch.randn(1, 3, 512, 512))
    print(y)
    # summary(net, (1, 3, 512, 512))


if __name__ == '__main__':
    test()
