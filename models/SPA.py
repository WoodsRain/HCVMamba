import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''
    def __init__(self, c, DW_Expand, dilation = 1):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        
        self.branch = nn.Sequential(
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel*4, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel, bias=True, dilation = dilation), # the dconv
                       nn.GroupNorm(4, self.dw_channel*4),
                       nn.GELU(),
                       nn.Conv2d(in_channels=self.dw_channel*4, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1) # the dconv
        )
    def forward(self, input):
        return self.branch(input)

class SimpleGate(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.wa1 = nn.Conv2d(c// 2, c// 2, 1, 1, 0)
        self.wa2 = nn.Conv2d(c// 2, c// 2, 1, 1, 0)
        self.sca = nn.Sequential(
                       nn.Conv2d(c// 2, c// 2, 3, 1, 1, groups = c//2, dilation =1),
                       nn.GELU(),
                       nn.Conv2d(c// 2, c// 2, 3, 1, 3, groups = c//2, dilation =3),
                       nn.GELU(),
                       nn.Conv2d(c// 2, c// 2, 3, 1, 5, groups = c//2, dilation =5),
                       nn.GELU(),
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(c// 2, c// 2, 1, 1, 0), 
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        z = (self.wa1(x1) + x2) - (x1 * self.wa2(x2))
        z = self.sca(z) * z
        return z
    
class DiSpAM(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dw_channel = DW_Expand * c 

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand = 1, dilation = dilation))
            
        assert len(dilations) == len(self.branches)

        self.com_g = SimpleGate(self.dw_channel)

        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)

        
    def forward(self, inp, adapter = None):

        y = inp
        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))

        z = 0
        for branch in self.branches:
            z += branch(x)    # bottleneck

        x = self.com_g(z)
        x = self.norm2(x)
        x = self.conv3(x)
        y = inp + x
        
        return y
