# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hypercomplex_layers import PHConv,PHMLinear
import numpy as np

__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4,n=2,cuda=False, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_reduce = PHConv(n, in_chs, reduced_chs, 1,cuda=cuda)
        self.act1 = act_layer(inplace=True)
        # self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.conv_expand = PHConv(n, reduced_chs, in_chs, 1,cuda=cuda)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU,n=2,cuda=False):
        super(ConvBnAct, self).__init__()
        # self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.conv = PHConv(n, in_chs, out_chs, kernel_size, kernel_size//2, stride,cuda=cuda)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,n=2,cuda=False):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            # nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            PHConv(n, inp, init_channels, kernel_size, kernel_size//2, stride,cuda=cuda),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,n=2,cuda=False):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                # nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                PHConv(n, in_chs, out_chs, 1, 0, 1,cuda=cuda),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=50, width=1.0, dropout=0.2,n=2,cuda=False):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.conv_head = PHConv(n, input_channel, output_channel, 1, 0, 1,cuda=cuda)
        self.act2 = nn.ReLU(inplace=True)
        # self.classifier = nn.Linear(output_channel, num_classes)
        self.classifier = PHMLinear(n, output_channel, num_classes,cuda=cuda)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, **kwargs)


#################
## PHM SeldNet ##
#################


class PHMSeldnet_augmented(nn.Module):
    def __init__(self, time_dim, freq_dim=256, input_channels=4, output_classes=14,
                 pool_size=[[8,2],[8,2],[2,2],[1,1]], cnn_filters=[64,128,256,512], pool_time=True,
                 rnn_size=256, n_rnn=3, fc_size=1024, dropout_perc=0.3, class_overlaps=3.,
                 verbose=False, n=2):
        super(PHMSeldnet_augmented, self).__init__()
        self.verbose = verbose
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        sed_output_size = output_classes * class_overlaps    #here 3 is the max number of simultaneus sounds from the same class
        doa_output_size = sed_output_size * 3   #here 3 is the number of spatial dimensions xyz
        if pool_time:
            self.time_pooled_size = int(time_dim / np.prod(np.array(pool_size), axis=0)[-1])
        else:
            self.time_pooled_size = time_dim
        #building CNN feature extractor
        conv_layers = []
        in_chans = input_channels
        for i, (p,c) in enumerate(zip(pool_size, cnn_filters)):
            curr_chans = c

            if pool_time:
                pool = [p[0],p[1]]
            else:
                pool = [p[0],1]
            conv_layers.append(
                nn.Sequential(
                    PHConv(n, in_chans, out_features=curr_chans,
                                kernel_size=3, stride=1, padding=1, cuda=False),  #padding 1 = same with kernel = 3
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.MaxPool2d(pool),
                    nn.Dropout(dropout_perc)))
            in_chans = curr_chans

        self.cnn = nn.Sequential(*conv_layers)

        self.rnn = nn.GRU(1024, rnn_size, num_layers=n_rnn, batch_first=True,
                          bidirectional=True, dropout=dropout_perc)

        self.sed = nn.Sequential(
#                     PHMLinear(n, rnn_size*2, fc_size),
                    nn.Linear(rnn_size*2, fc_size),
                    nn.ReLU(),
#                     PHMLinear(n, fc_size, fc_size),
                    nn.Linear(fc_size, fc_size),
                    nn.ReLU(),
#                     PHMLinear(n, fc_size, fc_size),
                    nn.Linear(fc_size, fc_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, sed_output_size),
                    nn.Sigmoid())

        # self.doa = nn.Sequential(
        #             nn.Linear(rnn_size*2, fc_size),
        #             nn.ReLU(),
        #             nn.Linear(fc_size, fc_size),
        #             nn.ReLU(),
        #             nn.Linear(fc_size, fc_size),
        #             nn.ReLU(),
        #             nn.Dropout(dropout_perc),
        #             nn.Linear(fc_size, doa_output_size),
        #             nn.Tanh())

    def forward(self, x):
        x = self.cnn(x)
        if self.verbose:
            print ('cnn out ', x.shape)    #target dim: [batch, n_cnn_filters, 2, time_frames]
        x = x.permute(0,3,1,2) #[batch, time, channels, freq]
        if self.verbose:
            print ('permuted: ', x.shape)    #target dim: [batch, time_frames, n_cnn_filters, 2]
        x = x.reshape(x.shape[0], self.time_pooled_size, -1)
        if self.verbose:
            print ('reshaped: ', x.shape)    #target dim: [batch, 2*n_cnn_filters]
        x, h = self.rnn(x)
        if self.verbose:
            print ('rnn out:  ', x.shape)    #target dim: [batch, 2*n_cnn_filters]
        sed = self.sed(x)
        # doa = self.doa(x)
        if self.verbose:
            print ('sed prediction:  ', sed.shape)  #target dim: [batch, time, sed_output_size]
            # print ('doa prediction: ', doa.shape)  #target dim: [batch, time, doa_output_size]

        return sed


import pretrainedmodels
model_name = 'resnet18'

basemodel = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
basemodel = nn.Sequential(*list(basemodel.children())[:-2])

class res18(nn.Module):
    def __init__(self):
        super(res18, self).__init__()
        self.features = basemodel
        if model_name == "resnet34" or model_name == "resnet18":
            num_ch = 512
        else:
            num_ch = 2048
        self.fc = nn.Conv2d(num_ch, 50, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.fc(x).squeeze(2).squeeze(2)
        return x



def count_parameters(model):
    return sum(p.numel() for p in model.cnn.parameters() if p.requires_grad)
if __name__=='__main__':
    # model = ghostnet()
    # print(model)
    # input = torch.randn(32,3,128,1723)
    # y = model(input)
    # print(y.size())
    from torchsummary import summary
    # model = PHMSeldnet_augmented(4800,pool_time=True, class_overlaps=3,input_channels=8,verbose=True, n=8)
    # summary(model, (8, 256, 4800))
    model = res18()
    summary(model, (3, 128, 1723))
