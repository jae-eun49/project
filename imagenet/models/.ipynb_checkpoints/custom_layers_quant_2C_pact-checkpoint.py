import torch
from torch import nn
from torch.autograd import Function
import pdb
from .module.quantize_2C import *
 
# Inherit from Function
class LinearFunctionCustom(Function):
    @staticmethod
    def forward(ctx, qinput, input, weight, bias=None):
        ctx.save_for_backward(qinput, input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
#         pdb.set_trace()
        qinput, input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
#         print('input',torch.unique(input).size(), input)
#         print('qinput',torch.unique(qinput).size(), qinput)
        
        if ctx.needs_input_grad[1]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output.t().mm(qinput)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)
            
        return None, grad_input, grad_weight, grad_bias

class LinearCustom(nn.Module):
    def __init__(self, input_features, output_features, bias=True, act_bits=8):
        super(LinearCustom, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # The same initialization as describe in the torch docs
        self.k = torch.sqrt(1/torch.tensor(input_features, dtype=torch.float32))
        self.weight.data = torch.randn((output_features, input_features), requires_grad=True) * self.k
        if bias is not None:
            self.bias.data.uniform_(-1.0*self.k, self.k)
            
        self.act_bits=act_bits
        self.quantize_input = QuantMeasure(num_bits=self.act_bits)

    def forward(self, input):
        if input[1] >= input[2] and input[1] < input[3]:
            qinput = self.quantize_input(input[0])
        else:
            qinput = input[0]
        return LinearFunctionCustom.apply(qinput, input[0], self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    

class Conv2d_imagenet(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d_imagenet, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits_input = 2
        self.num_bits_weight = 8
        self.quantize_input = QuantMeasure(self.num_bits_input, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        
    def forward(self, input):
#         pdb.set_trace()
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams( self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
            
        return F.conv2d(qinput, qweight, self.bias, self.stride,self.padding, self.dilation, self.groups)



class Conv2DFunctionCustom(Function):
    @staticmethod
    def forward(ctx, qinput, qweight, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        
        ctx.save_for_backward(qinput, qweight, bias)
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
#         print(weight)
        output = torch.nn.functional.conv2d(qinput, qweight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

        


        return output

    @staticmethod
    def backward(ctx, grad_output):
        #import pdb
        #import pydevd
        #pydevd.settrace(suspend=True, trace_only_current_thread=True)
        input, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        
        grad_input = grad_weight = grad_qweight = grad_qinput = grad_bias = grad_stride = grad_padding = grad_dilation = grad_groups = None
        
        if ctx.needs_input_grad[1]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)

#         if ctx.needs_input_grad[3]:
#             print('????')
#             grad_bias = grad_output.sum(0, 2, 3).squeeze(0)  # todo: double check
#             print('4')
            
#         grad_test1 = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
#         grad_test2 = torch.nn.grad.conv2d_weight(qinput, weight.shape, grad_output, stride, padding, dilation, groups)
#         print('grad_input', grad_test1[0][0])
#         print('grad_qinput', grad_test2[0][0])
#         print('grad_weight', grad_weight[0][0])

        return grad_qinput, grad_qweight, grad_input, grad_weight, grad_bias, grad_stride, grad_padding, grad_dilation, grad_groups


class Conv2DCustom(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2DCustom, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.num_bits_input = 2
        self.num_bits_weight = 8
        self.quantize_input = QuantMeasure(self.num_bits_input, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))

    def forward(self, x):
#         print('x', x.size())
        qinput = self.quantize_input(x)
        weight_qparams = calculate_qparams( self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
        
        
        return Conv2d_imagenet.apply(qinput, qweight, x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size
        )

