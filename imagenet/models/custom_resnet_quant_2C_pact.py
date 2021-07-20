'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
from torch.nn import Conv2d as Conv2dNormal
from torch.nn import Linear as LinearNormal
from .custom_layers_quant_2C_pact import Conv2d_imagenet
from .custom_layers_quant_2C_pact import LinearCustom
import torch.nn.functional as F
from .module.quantize_2C import *

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, alpha):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        
        relu_input2 = 0.5*(torch.abs(input)-torch.abs(input-alpha)+alpha)
        relu_input = nn.functional.relu(relu_input2)
        
        return relu_input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_alpha = None
        
        grad_input = grad_output.clone()
        grad_input = torch.where(input < 0, torch.zeros_like(grad_input), grad_input)
        grad_input = torch.where(input > alpha, torch.zeros_like(grad_input), grad_input)
        
        grad_temp = grad_output.clone()
        grad_temp = torch.where(input < alpha, torch.zeros_like(grad_temp), grad_temp)
        grad_alpha = torch.sum(grad_temp)
#         print(grad_alpha)
        
        return grad_input, grad_alpha
    
class PACT(nn.ReLU):
    def __init__(self):
        super(PACT, self).__init__()
        
        self.alpha = nn.Parameter(torch.tensor(10, dtype=torch.float32))

    def forward(self, x):
        
        return MyReLU.apply(x, self.alpha)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = PACT()
        self.relu2 = PACT()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        
#         print('1 : ', out.size())
        out = self.relu1(out)
#         print('2 : ', out.size())
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
# #         self.conv1 = QConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, num_bits=8, num_bits_weight=8)
#         self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.relu1 = PACT()
#         self.relu2 = PACT()

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#             self.shortcut_en = 1
#         else:
#             self.shortcut_en = 0

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2((out,x[1],x[2])))
#         if self.shortcut_en == 1:
#             out += self.shortcut((x[0],x[1],x[2]))
#         else:
#             out += self.shortcut(x[0])
#         out = self.relu2(out)
#         return (out, x[1], x[2])


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, act_bits=8):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False, act_bits=act_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, act_bits=act_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False, act_bits=act_bits)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = PACT()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, act_bits=act_bits),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
    
class ResNet_cifar10(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar10, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.relu = PACT()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
#         print('1', out.size())
        out1 = self.relu(out)
#         print('2', out1.size())
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = F.avg_pool2d(out4, 8)
        out6 = out5.view(out5.size(0), -1)
        out7 = self.linear(out6)
        return out7


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.relu = PACT()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3,2)
#         print(out.size())
        out = self.relu(self.bn1(out))
#         print(out.size())
        out = self.layer1(out)
#         print(out.size())
        out = self.layer2(out)
#         print(out.size())
        out = self.layer3(out)
#         print(out.size())
        out = self.layer4(out)
#         print(out.size())
        out = F.avg_pool2d(out, 7)
#         print(out.size())
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


Conv2d = None
Linear = None

def ResNet18(normal=False):
    global Conv2d
    global Linear
#     Linear = LinearNormal
    if normal:
        Conv2d = Conv2dNormal
        Linear = LinearNormal
    else:
        Conv2d = Conv2d_imagenet
        Linear = LinearNormal
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet20(normal=False):
    global Conv2d
    global Linear
#     Linear = LinearNormal
    if normal:
        Conv2d = Conv2dNormal
        Linear = LinearNormal
    else:
        Conv2d = Conv2DCustom
        Linear = LinearCustom
    return ResNet_cifar10(BasicBlock, [3,3,3])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
