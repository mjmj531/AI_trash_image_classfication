import torch
import torch.nn as nn

# ResNet定义
class ResNet_mine(nn.Module):
    def __init__(self, block, layers, num_classes=10, use_dropout=False):
        super(ResNet_mine, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Dropout层
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)  # 50%的丢弃率 ############

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers) # 接受多个层作为输入

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        
        # 全局平均池化
        x = self.avgpool(x)
        # Flatten
        x = torch.flatten(x, 1)
        # # 原始论文中没有使用dropout层
        # # # 这里尝试以下dropout层的效果
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x


# 定义 ResNet BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) # 在conv层和ReLU层之间进行BN
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample 
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 保证原始输入x的size与主分支卷积后的输出size相同，从而叠加
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接结构
        out += identity
        out = self.relu(out)

        return out

# 定义 Bottleneck 模块（ResNet50及以上）

class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数是输入通道数的 4 倍

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 第一层：1x1 卷积，用于减少通道数
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二层：3x3 卷积，用于提取特征
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 第三层：1x1 卷积，用于恢复通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNet18 层配置
def resnet18_handwriting(num_classes=10, use_dropout=False):
    # 共4个残差层
    return ResNet_mine(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, use_dropout=use_dropout)

# ResNet34 层配置
def resnet34_handwriting(num_classes=10, use_dropout=False):
    # 调整层数配置为 [3, 4, 6, 3]
    return ResNet_mine(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, use_dropout=use_dropout)

# ResNet50 层配置
def resnet50_handwriting(num_classes=10, use_dropout=False):
    # 调整层数配置为 [3, 4, 6, 3]
    return ResNet_mine(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, use_dropout=use_dropout)
