import torch
import torch.nn as nn
from collections import OrderedDict
import math


# BasicBlock ： 残差块，此残差块残差边上并没有卷积，故通过此残差块并不改变特征图的通道数
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    # 参数layers记录了各个块的个数
    def __init__(self, layers):
        super().__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        # 暂时用处不明
        self.layers_out_filters = [64, 128, 256, 512, 1024]
        # 下面主要是对模型中的参数进行初始化的操作
        # 其中self.modules()会返回这个模型中的所有层
        for m in self.modules():
            # isinstance有两个参数，一个为对象，一个为类型，其功能是判断这个对象是不是某类型
            # 下面的m就是一个具体的卷积层的对象
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []

        # 将几个元组插入到列表里面
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


# def darknet53(pretrained, **kwargs):
#     model = DarkNet([1, 2, 8, 8, 4])
#     if pretrained:
#         if isinstance(pretrained, str):
#             model.load_state_dict(torch.load(pretrained))
#         else:
#             raise Exception("darknet request a pretrained path.got[{}]".format(pretrained))
#
#     return model
model = DarkNet([1, 2, 8, 8, 4])
input = torch.randn(1, 3, 416, 416)
model.eval()

torch.onnx.export(model, input, "yolov3-darknet53.onnx", opset_version=12)
