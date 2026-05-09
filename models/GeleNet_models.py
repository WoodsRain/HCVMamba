import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout
from typing import Optional
import math

from typing import List, Callable
from torch import Tensor

# out = channel_shuffle(out, 2)
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 8, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SWSAM(nn.Module):
    def __init__(self, channels, factor=4):
        super(SWSAM, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.bn = nn.BatchNorm2d(channels // self.groups) 
        self.bn1 = nn.BatchNorm2d(channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv7x7 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=7, stride=1, padding=3)
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.convs = nn.Sequential(nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1))
                        

    def forward(self, x):
        b, c, h, w = x.size()
        x = channel_shuffle(x, 4)
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)  # b*g,c//g,h,1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # b*g,c//g,w,1
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # b*g,c//g, h + w,1
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x_g = self.convs(group_x)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x3 = self.conv7x7(group_x)
        x4 = self.bn(group_x * x_g.sigmoid())
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # b*g, 1, c//g
        x12 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # b*g, 1, c//g
        x22 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x41 = self.softmax(self.agp(x3).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # b*g, 1, c//g
        x42 = x3.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x31 = self.softmax(self.agp(x4).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # b*g, 1, c//g
        x32 = x4.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw             
        
        nor_weights = F.softmax(self.weight, dim=0)
        weights = (torch.matmul(x11, x22) * nor_weights[0] + torch.matmul(x21, x32) * nor_weights[1] + torch.matmul(x31, x42) * nor_weights[2] + torch.matmul(x41, x12) * nor_weights[3]).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
    
    
"""    
# SWSAM: Shuffle Weighted Spatial Attention Module
class SWSAM(nn.Module):    # Fusion Attention
    def __init__(self, channel=32): # group=8, branch=4, group x branch = channel
        super(SWSAM, self).__init__()

        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.sa_fusion = nn.Sequential(BasicConv2d(1, 1, 3, padding=1),
                                       nn.Sigmoid()
                                       )

    def forward(self, x):
        x = channel_shuffle(x, 4)
        x1, x2, x3, x4 = torch.split(x, 8, dim = 1)
        s1 = self.SA1(x1)
        s2 = self.SA1(x2)
        s3 = self.SA1(x3)
        s4 = self.SA1(x4)
        nor_weights = F.softmax(self.weight, dim=0)
        s_all = s1 * nor_weights[0] + s2 * nor_weights[1] + s3 * nor_weights[2] + s4 * nor_weights[3]
        x_out = self.sa_fusion(s_all) * x + x

        return x_out
"""
    
    
class DirectionalConvUnit(nn.Module):
    def __init__(self, channel):
        super(DirectionalConvUnit, self).__init__()

        self.h_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))
        self.w_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        # leading diagonal
        self.dia19_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        # reverse diagonal
        self.dia37_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))

    def forward(self, x):

        x1 = self.h_conv(x)
        x2 = self.w_conv(x)
        x3 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))
        x4 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)

        return x

    # Code from "CoANet- Connectivity Attention Network for Road Extraction From Satellite Imagery", and we modified the code
    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x.permute(0, 1, 3, 2)

class CosineSimilarity(nn.Module):
 
    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

# Knowledge Transfer Module
class KTM(nn.Module):
    def __init__(self, channel=32, pre_norm=False):
        super(KTM, self).__init__()

        self.query_conv = nn.Conv2d(channel, channel, kernel_size=1) 
        self.key_conv = nn.Conv2d(channel, channel, kernel_size=1) 
        
        self.value_conv_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_3 = nn.Conv2d(channel, channel, kernel_size=1)
        
        self.gamma_2 = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.gamma_3 = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.gamma_4 = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.gamma_5 = nn.Parameter(torch.ones(1, channel, 1, 1))
        
        self.softmax = Softmax(dim=-1)

        # following DANet
        self.conv_2 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 3, padding=1)
                                    )
        self.conv_3 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 3, padding=1)
                                    ) 
        
        self.conv_out = nn.Sequential(
                                      nn.Dropout2d(0.1, False),
                                      nn.Conv2d(channel, channel, 1)
                                      )
        self.conv_q =  nn.Conv2d(channel, channel, 1)
        self.conv_k = nn.Conv2d(channel, channel, 1)
        self.cosine = CosineSimilarity()


    def forward(self, x2, x3, pos): # V
    
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        bs, c, h, w = x2.size()
        
        x_sum = x2 + x3 # Q use dw conv to replace
        x_mul = x2 * x3 # K use dw conv to replace

        m_batchsize, C, height, width = x_sum.size()
        proj_query = self.query_conv(x_sum + pos).view(m_batchsize, -1, width * height).permute(0, 2, 1) # bs hw c
        proj_key = self.key_conv(x_mul + pos).view(m_batchsize, -1, width * height) # bs c hw
        
        # scale = c ** -0.5
        
        value_query = torch.sqrt((proj_query ** 2).sum(2).unsqueeze(2))  # bs hw 1
        value_key = torch.sqrt((proj_key ** 2).sum(1).unsqueeze(1))  # bs 1 hw    
        
        energy = torch.bmm(proj_query, proj_key) # bs hw hw
        scale = torch.bmm(value_query, value_key)
        attention = energy / scale  # bs hw hw

        proj_value_2 = self.value_conv_2(x2).view(m_batchsize, -1, width * height) # bs c hw
        proj_value_3 = self.value_conv_3(x3).view(m_batchsize, -1, width * height) # bs c hw

        out_2 = torch.bmm(proj_value_2, attention.permute(0, 2, 1)) # bs c hw
        PatchCosine2 = self.cosine(out_2.transpose(1,2), x2.flatten(2).transpose(1,2)).unsqueeze(1) # bs 1 hw
        out_2 = out_2 * PatchCosine2 + out_2 
        out_2 = out_2.view(m_batchsize, C, height, width)
        
        out_3 = torch.bmm(proj_value_3, attention.permute(0, 2, 1)) # bs c hw
        PatchCosine3 = self.cosine(out_3.transpose(1,2), x3.flatten(2).transpose(1,2)).unsqueeze(1) # bs 1 hw 
        out_3 = out_3 * PatchCosine3 + out_3 # + proj_query.transpose(1,2)
        out_3 = out_3.view(m_batchsize, C, height, width)
        
        out_2 = self.conv_2(self.gamma_2 * out_2 + x2) # No change  
        out_3 = self.conv_3(self.gamma_3 * out_3 + x3) # No change

        x_out = self.conv_out(self.gamma_4 * out_2 + self.gamma_5 * out_3) # out_2 + out_3

        return x_out


class PDecoder(nn.Module):
    def __init__(self, channel):
        super(PDecoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        
        self.conv_upsample3_1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3_2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3_3 = BasicConv2d(channel, channel, 3, padding=1)
        
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x2_5, x3): # x1: 32x11x11, x2: 32x22x22, x2_5: 32x44x44 x3: 32x88x88,
        x1_1 = x1 # 32x11x11
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 # 32x22x22
        x2_5_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x2_5
        x3_1 = self.conv_upsample3_1(self.upsample(self.upsample(self.upsample(x1)))) \
               * self.conv_upsample3_2(self.upsample(self.upsample(x2))) * self.conv_upsample3_3(self.upsample(x2_5)) * x3 # 32x88x88

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1) # 32x22x22
        x2_2 = self.conv_concat2(x2_2)
        
        x2_5_2 = torch.cat((x2_5_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x2_5_2 = self.conv_concat3(x2_5_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample6(self.upsample(x2_5_2))), 1) # 32x88x88
        x3_2 = self.conv_concat4(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x) # 1x88x88

        return x


class GeleNet(nn.Module):
    def __init__(self, channel=32, pre_norm=False):
        super(GeleNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # input 3x352x352
        self.ChannelNormalization_1 = BasicConv2d(64, channel, 3, 1, 1)  # 64x88x88->32x88x88
        self.ChannelNormalization_1_1 = BasicConv2d(channel, channel, 3, 2, 1)
        self.ChannelNormalization_2_1 = BasicConv2d(128, channel, 3, 1, 1)
        self.ChannelNormalization_2 = BasicConv2d(channel, channel, 3, 2, 1) # 128x44x44->32x22x22
        self.ChannelNormalization_3 = BasicConv2d(320, channel, 3, 1, 1) # 320x22x22->32x22x22
        self.ChannelNormalization_3_1 = BasicConv2d(channel, channel, 3, 2, 1)
        self.ChannelNormalization_4 = BasicConv2d(512, channel, 3, 1, 1) # 512x11x11->32x11x11

        
        self.SWSAM_4 = SWSAM(channel)
        
        # D-SWSAM for x1_nor
        self.dirConv = DirectionalConvUnit(channel)
        self.DSWSAM_1 = SWSAM(channel) # group x branch = channel

        # KTM for x2_nor and x3_nor
        self.KTM_12 = KTM(channel)
        self.KTM_23 = KTM(channel)
        self.KTM_34 = KTM(channel)     

        """
        self.num_layers = 8
        self.features_cross_attention_layers = nn.ModuleList()
        self.features_ffn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.features_cross_attention_layers.append(nn.MultiheadAttention(channel, 4, dropout=0.0))
            self.features_ffn_layers.append(
                    FFNLayer(
                        d_model=channel,
                        dim_feedforward=channel*8,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    ))  
        """
        
        self.pe_layer = PositionEmbeddingSine(channel//2)
        self.PDecoder = PDecoder(channel)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11

        x1_nor = self.ChannelNormalization_1(x1) # 32x88x88 
        x1_ori = self.dirConv(x1_nor)
        x1_DSWSAM_1 = self.DSWSAM_1(x1_ori) # 32x88x88
        
        x1_1_nor = self.ChannelNormalization_1_1(x1_DSWSAM_1) # 32x44x44
        x2_1_nor = self.ChannelNormalization_2_1(x2) # 32x44x44
        pos1 = self.pe_layer(x1_1_nor)
        x12_KTM = self.KTM_12(x1_1_nor, x2_1_nor, pos1)
        
        x2_nor = self.ChannelNormalization_2(x12_KTM) # 32x22x22
        x3_nor = self.ChannelNormalization_3(x3) # 32x22x22
        pos2 = self.pe_layer(x2_nor)
        x23_KTM = self.KTM_23(x2_nor, x3_nor, pos2)
        
        x3_1_nor = self.ChannelNormalization_3_1(x23_KTM)
        x4_nor = self.ChannelNormalization_4(x4)
        pos3 = self.pe_layer(x3_1_nor)
        x34_KTM = self.KTM_34(x3_1_nor, x4_nor, pos3)
        
        x4_SWSAM_4 = self.SWSAM_4(x34_KTM)
        
        """
        bs, c, h, w = x4_SWSAM_4.size()
        
        res = [x4_SWSAM_4, x23_KTM, x12_KTM, x1_DSWSAM_1]
                
        k = []
        v = []
        for i in range(len(res)):
            k.append(res[i].flatten(2).permute(2,0,1) + self.pe_layer(res[i], None).flatten(2).permute(2,0,1))
            v.append(res[i].flatten(2).permute(2,0,1))
        
        pos_query = self.pe_layer(x4_SWSAM_4, None).flatten(2).permute(2,0,1)
        Query = x4_SWSAM_4.flatten(2).permute(2,0,1) + pos_query
                
        for i in range(self.num_layers):
            level_index = i % 4
            output, _ = self.features_cross_attention_layers[i](query = Query, key = k[level_index], value = v[level_index])
            output = self.features_ffn_layers[i](output)
            if i != self.num_layers - 1:
                Query = output + pos_query
            else:
                Query = output.permute(1,2,0).reshape(bs, c, h, w)
            
        x4_SWSAM_4 = Query   
        """
        
        prediction = self.upsample_4(self.PDecoder(x4_SWSAM_4, x23_KTM, x12_KTM, x1_DSWSAM_1))  # x4_SWSAM_4, x1_DSWSAM_1

        return prediction, self.sigmoid(prediction)

    
class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)    

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")    
     
"""
class GeleNet(nn.Module):
    def __init__(self, channel=32):
        super(GeleNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # input 3x352x352
        self.ChannelNormalization_1 = BasicConv2d(64, channel, 3, 1, 1)  # 64x88x88->32x88x88
        self.ChannelNormalization_1_1 = BasicConv2d(32, channel, 3, 2, 1)
        self.ChannelNormalization_2_1 = BasicConv2d(128, channel, 3, 1, 1)
        self.ChannelNormalization_2 = BasicConv2d(32, channel, 3, 2, 1) # 128x44x44->32x22x22
        self.ChannelNormalization_3 = BasicConv2d(320, channel, 3, 1, 1) # 320x22x22->32x22x22
        self.ChannelNormalization_3_1 = BasicConv2d(32, channel, 3, 2, 1)
        self.ChannelNormalization_4 = BasicConv2d(512, channel, 3, 1, 1) # 512x11x11->32x11x11

        
        self.SWSAM_4 = SWSAM(channel)
        
        # D-SWSAM for x1_nor
        self.dirConv = DirectionalConvUnit(channel)
        self.DSWSAM_1 = SWSAM(channel) # group x branch = channel

        # KTM for x2_nor and x3_nor
        self.KTM_12 = KTM(channel)
        self.KTM_23 = KTM(channel)
        self.KTM_34 = KTM(channel)

        self.PDecoder = PDecoder(channel)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11

        x1_nor = self.ChannelNormalization_1(x1) # 32x88x88 
        x1_ori = self.dirConv(x1_nor)
        x1_DSWSAM_1 = self.DSWSAM_1(x1_ori) # 32x88x88
        
        x1_1_nor = self.ChannelNormalization_1_1(x1_DSWSAM_1) # 32x44x44
        x2_1_nor = self.ChannelNormalization_2_1(x2) # 32x44x44
        x12_KTM = self.KTM_12(x1_1_nor, x2_1_nor)
        
        x2_nor = self.ChannelNormalization_2(x12_KTM) # 32x22x22
        x3_nor = self.ChannelNormalization_3(x3) # 32x22x22
        x23_KTM = self.KTM_23(x2_nor, x3_nor)
        
        x3_1_nor = self.ChannelNormalization_3_1(x23_KTM)
        x4_nor = self.ChannelNormalization_4(x4)
        x34_KTM = self.KTM_34(x3_1_nor, x4_nor)
        
        x4_SWSAM_4 = self.SWSAM_4(x34_KTM)
            
        prediction = self.upsample_4(self.PDecoder(x4_SWSAM_4, x23_KTM, x12_KTM,  x1_DSWSAM_1))  # x4_SWSAM_4, x1_DSWSAM_1

        return prediction, self.sigmoid(prediction)
"""
                
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        #dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor'))
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)