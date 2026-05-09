import torch.nn as nn
import torch
import torch.nn.functional as F
from models.GBC import GBC, BottConv
from models.DySample import DySample
from models.SPA import DiSpAM
import math

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
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

class CosineSimilarity(nn.Module):
 
    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

# Knowledge Transfer Module
class KTM(nn.Module):
    def __init__(self, channel=32, pre_norm=False):
        super(KTM, self).__init__()
        
        self.channel = channel

        self.query_conv = nn.Conv2d(channel, channel, kernel_size=1) 
        self.key_conv = nn.Conv2d(channel, channel, kernel_size=1) 
        
        self.value_conv_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_3 = nn.Conv2d(channel, channel, kernel_size=1)
        
        self.prior_conv_2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.prior_conv_3 = nn.Conv2d(channel, channel, 3, 1, 1)
        
        self.gamma_2 = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.gamma_3 = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.gamma_4 = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.gamma_5 = nn.Parameter(torch.ones(1, channel, 1, 1))
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(8)
        self.interact_mlp = nn.Sequential(
            nn.Linear(64 * 2, 64 * 4),
            nn.LayerNorm(64 * 4),
            nn.GELU(),
            nn.Dropout2d(0.1, False),
            nn.Linear(64 * 4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout2d(0.1, False),
            nn.Linear(64, 1)  
        )

        # following DANet
        self.conv_2 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.GELU(),
                                    nn.Dropout2d(0.1, False),
                                    BasicConv2d(channel, channel, 3, padding=1),
                                    nn.GELU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 3, padding=1)
                                    )
        self.conv_3 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.GELU(),
                                    nn.Dropout2d(0.1, False),
                                    BasicConv2d(channel, channel, 3, padding=1),
                                    nn.GELU(),
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
        self.pe_layer = PositionEmbeddingSine(channel//2)


    def forward(self, x2, x3): # V
    
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        bs, c, h, w = x2.size()
        pos = self.pe_layer(x2)
        
        x_sum = x2 + x3 # Q use dw conv to replace
        x_mul = x2 * x3 # K use dw conv to replace

        m_batchsize, C, height, width = x_sum.size()
        
        x_pool = self.avg_pool(x_sum).view(m_batchsize, -1, 8 * 8)  # bs c 64
        proj_query = self.query_conv(x_sum + pos).view(m_batchsize, -1, width * height) # bs c hw
        proj_key = self.key_conv(x_mul + pos).view(m_batchsize, -1, width * height).permute(0, 2, 1) # bs hw c
        
        scale = c ** -0.5 
        
        energy = torch.bmm(proj_query, proj_key) # bs c c
        attention = energy / scale  # bs c c
        
        proj_value_2 = self.prior_conv_2(self.value_conv_2(x2)).view(m_batchsize, -1, width * height) # bs c hw
        proj_value_3 = self.prior_conv_3(self.value_conv_3(x3)).view(m_batchsize, -1, width * height) # bs c hw
        
        # print(proj_value_2.size(),proj_query.size()) 
        prior2 = (self.softmax(proj_value_2) * proj_query).sum(2).unsqueeze(2)  # b c 1
        prior3 = (self.softmax(proj_value_3).transpose(-1, -2) * proj_key).sum(1).unsqueeze(1)  # b 1 c
        
        prior2_1 = (torch.sigmoid(prior2.unsqueeze(3)) * x_pool.unsqueeze(2)).expand(-1, -1, self.channel, 64)   # b c c 64
        prior3_1 = (torch.sigmoid(prior3.unsqueeze(3)) * x_pool.unsqueeze(1)).expand(-1, self.channel, -1, 64)   # b c c 64
        # print(prior2_1.size(), prior3_1.size())
        pair_feat = torch.cat([prior2_1, prior3_1], dim=-1)   # b c c 64*2
        pair_score = self.interact_mlp(pair_feat).squeeze(-1)  # b c c
        
        PatchCosine0 = self.cosine(pair_score, attention).unsqueeze(1)   # b 1 c
        PatchCosine1 = self.cosine(pair_score.transpose(1,2), attention.transpose(1,2)).unsqueeze(1).transpose(1,2)    # b c 1
        
        attention = self.softmax(attention + pair_score)      

        out_2 = torch.bmm(attention, proj_value_2) # bs c hw
        PatchCosine2 = self.cosine(out_2.transpose(1,2), x2.flatten(2).transpose(1,2)).unsqueeze(1) # bs 1 hw
        out_2 = out_2 * PatchCosine2 + out_2 
        out_2 = torch.bmm(attention * PatchCosine0 * PatchCosine1, out_2)
        out_2 = out_2.view(m_batchsize, C, height, width)
        
        out_3 = torch.bmm(attention, proj_value_3) # bs c hw
        PatchCosine3 = self.cosine(out_3.transpose(1,2), x3.flatten(2).transpose(1,2)).unsqueeze(1) # bs 1 hw 
        out_3 = out_3 * PatchCosine3 + out_3 # + proj_query.transpose(1,2)
        out_3 = torch.bmm(attention * PatchCosine0 * PatchCosine1, out_3)
        out_3 = out_3.view(m_batchsize, C, height, width)
        
        out_2 = self.conv_2(self.gamma_2 * out_2 + x2) # No change  
        out_3 = self.conv_3(self.gamma_3 * out_3 + x3) # No change

        x_out = self.conv_out(self.gamma_4 * out_2 + self.gamma_5 * out_3) # out_2 + out_3

        return x_out

"""
class CoFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3,
            stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
            stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3,
            stride=1, padding=1)
        self.relu = nn.ReLU()

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)


    def forward(self, x):
        fusecat = x # torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(fusecat)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)
        

        return ((fusecat * attn).sum(1)).unsqueeze(1)
"""        
        
class COI(nn.Module):
    def __init__(self, inc, k=3, p=1):
        super().__init__()
        self.outc = inc
        self.dw = nn.Conv2d(inc, self.outc, kernel_size=k, padding=p, groups=inc)
        self.conv1_1 = nn.Conv2d(inc, self.outc, kernel_size=1, stride=1)
        self.bn1 = nn.GroupNorm(4, self.outc) # nn.BatchNorm2d(self.outc)
        self.bn2 = nn.GroupNorm(4, self.outc) # nn.BatchNorm2d(self.outc)
        self.bn3 = nn.GroupNorm(4, self.outc) # nn.BatchNorm2d(self.outc)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = self.bn1(x)

        x_dw = self.bn2(self.dw(x))

        x_conv1_1 = self.bn3(self.conv1_1(x))

        return self.act(shortcut + x_dw + x_conv1_1)


class MHMC(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=True, proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()
        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        for i in range(self.ca_num_heads):
            local_attn = DiSpAM(dim // self.ca_num_heads)
            local_conv = nn.Sequential(nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2), padding=(1 + i), stride=1, groups=dim // self.ca_num_heads),
                                                       nn.GELU(),
                                                       nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, 1, 1, 0),
                                                       nn.SiLU())
            setattr(self, f"local_attn_{i + 1}", local_attn)
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                              groups=self.split_groups)
        self.bn = nn.GroupNorm(4,  dim * expand_ratio) # nn.BatchNorm2d(dim * expand_ratio)
        self.conv1 = nn.Conv2d(self.split_groups//2, self.split_groups//2, 1, 1, 0)
        self.conv2 = nn.Conv2d(self.split_groups//2, self.split_groups//2, 1, 1, 0)
        self.gelu = nn.GELU()
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        v = self.v(x)

        s = (self.q(x) * self.s(x)).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)
        for i in range(self.ca_num_heads):
            local_attn = getattr(self, f"local_attn_{i + 1}")
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]
            s_a = local_attn(s_i)   # .reshape(B, self.split_groups, -1, H, W)
            s_a_p = self.gelu(s_a)
            s_a_n = self.gelu(-s_a)
            s_i = local_conv(s_i)  # .reshape(B, self.split_groups, -1, H, W)
            s_i_p1, s_i_p2 = torch.chunk(s_i * s_a_p, 2, dim=1)
            s_i_n1, s_i_n2 = torch.chunk(s_i * s_a_n, 2, dim=1)
            s_i_1 = self.conv1(s_i_p1 + s_i_n2)
            s_i_2 = self.conv2(s_i_p2 + s_i_n1)
            s_i = torch.cat([s_i_1, s_i_2], dim=1).reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        s_out = s_out.reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)

        x = s_out * v

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MAFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.outc = inc

        self.pre_att = nn.Sequential(
            nn.Conv2d(inc * 4, inc * 2, kernel_size=3, padding=1, groups=inc * 2),
            nn.GroupNorm(4, inc * 2),   # nn.BatchNorm2d(inc * 2),
            nn.GELU(),
            nn.Conv2d(inc * 2, inc*4, kernel_size=1),
            nn.GroupNorm(4, inc * 4), 
            nn.GELU()
        )

        self.attention = MHMC(dim=inc*4)

        self.coi = COI(inc*4)

        self.pw = nn.Sequential(
            nn.Conv2d(in_channels=inc*4, out_channels=inc*4, kernel_size=1, stride=1),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_cat = torch.cat(x, dim=1)

        # DW-BN-GELU
        x_pre = self.pre_att(x_cat)

        # MHMC
        x_reshape = x_pre.flatten(2).permute(0, 2, 1)
        attention = self.attention(x_reshape, H, W)
        attention = attention.permute(0, 2, 1).reshape(B, 4 * C, H, W)

        x_conv = self.coi(attention)
        x_conv = self.softmax(self.pw(x_conv))

        x_res = (x_conv * attention).sum(1).unsqueeze(1)
        return x_res

class MFS(nn.Module):
    def __init__(self, embedding_dim):
        super(MFS, self).__init__()

        self.embedding_dim = embedding_dim
        self.linear_c4 = MLP(input_dim=128, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=64, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=32, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=16, embed_dim=embedding_dim)
        
        self.CoFusion = MAFM(embedding_dim) # CoFusion(4 * embedding_dim, 4 * embedding_dim)
        # self.CoFusion = nn.Conv2d(32, 1, 1, 1, 0)
        # self.GBC_C = GBC(embedding_dim*4)
        # self.linear_fuse = BottConv(embedding_dim*4, embedding_dim, embedding_dim//8, kernel_size=1, padding=0, stride=1)
        
        self.c4_pred = nn.Sequential(
            BottConv(embedding_dim, 1, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        
        self.c3_pred = nn.Sequential(
            BottConv(embedding_dim, 1, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        
        self.c2_pred = nn.Sequential(
            BottConv(embedding_dim, 1, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        
        self.c1_pred = nn.Sequential(
            BottConv(embedding_dim, 1, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        
        
        # self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)
        # self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        # self.dropout = nn.Dropout(p=0.1)

        self.DySample_C_2 = DySample(embedding_dim, scale=2)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_8 = DySample(embedding_dim, scale=8)
        
        self.trans4_3 = KTM(embedding_dim)
        self.trans3_2 = KTM(embedding_dim)
        self.trans2_1 = KTM(embedding_dim)

    def forward(self, inputs):
        c4, c3, c2, c1 = inputs
        # print(c4.size(), c3.size(), c2.size(), c1.size())  128x64x64 64x128x128  32x256x256  16x512x512  
        b, c, h, w = c4.shape
        out_c4 = self.linear_c4(c4.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c4, out_c4_3 = self.DySample_C_8(out_c4), self.DySample_C_2(out_c4)

        b, c, h, w = c3.shape
        out_c3 = self.linear_c3(c3.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c3 = self.trans4_3(out_c4_3, out_c3)
        out_c3, out_c3_2 = self.DySample_C_4(out_c3), self.DySample_C_2(out_c3)  #  

        b, c, h, w = c2.shape
        out_c2 = self.linear_c2(c2.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c2 = self.trans3_2(out_c3_2, out_c2)
        out_c2 = self.DySample_C_2(out_c2)

        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(c1.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c1 = self.trans2_1(out_c2, out_c1)
        
        x_c4 = self.c4_pred(out_c4)
        x_c3 = self.c3_pred(out_c3)
        x_c2 = self.c2_pred(out_c2)
        x_c1 = self.c1_pred(out_c1)        
        
        x = self.CoFusion([out_c4, out_c3, out_c2, out_c1])

        return x_c4, x_c3, x_c2, x_c1, x
        
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
