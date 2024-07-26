import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F


class _quantize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step_size, half_lvls): # quantize into fixed-point int
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())
        
        # print (f"output in quantize = {output}")

        output = torch.round(output / ctx.step_size) # quantize into fixed-point int
        # print (f"output in quantize after divide step size = {output}")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step_size

        return grad_input, None, None

# ZX: quantize function definition
# ZX: tensor.apply(): Applies the function callable to each element in the tensor, 
# replacing each element with the value returned by callable.
quantize = _quantize_func.apply


class quan_Conv2d(nn.Conv2d): # ZX: instance of a quantized convolution layer
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(quan_Conv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        self.N_bits = 8 # ZX: N_bits is defined here
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place change MSB to negative

    def forward(self, input):
        if self.inf_with_weight:
            return F.conv2d(input, self.weight * self.step_size, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            return F.conv2d(input, weight_quan, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = (self.weight.abs().max() / self.half_lvls)*1.5

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True


class quan_Linear(nn.Linear): # ZX: instance of a quantized linear layer
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)
        
        # print (f"before quantization weight in quan_Linear:")
        # print (self.weight)
 
        

        self.N_bits = 8 # ZX: weight length after quantization = Nq
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place reverse
        # why reverse the LSB?
        
        
        # print (f"after quantization initialization weight in quan_Linear:")
        # print (self.weight)


    def forward(self, input): # the 1st time of forward will trigger reset stepsize
        if self.inf_with_weight: # after weight reset
            return F.linear(input, self.weight * self.step_size, self.bias)
        else: # before weight reset
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size # * step_size -> weight normalization to keep all quantized weight within range of 2^(Nq-1)-1 (no overflow)
            return F.linear(input, weight_quan, self.bias) # if 1st time trigger forward, quantize weights before running forward

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = (self.weight.abs().max() / self.half_lvls)*1.5

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replaceblocks[0].mlp.fc2. the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True
