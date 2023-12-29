import pretrainedmodels
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
import os


#残差网络预训练模型
class resnet(nn.Module):
    def __init__(self, model_name):
        super(resnet, self).__init__()
        basemodel = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        basemodel = nn.Sequential(*list(basemodel.children())[:-2])
        self.features = basemodel
        if model_name == "resnet34" or model_name == "resnet18":
            num_ch = 512
        else:
            num_ch = 2048
        self.pool = nn.AdaptiveAvgPool2d(1)#(1,1)
        self.fc = nn.Conv2d(num_ch, 50, 1)#与全连接层类似，是1*1的卷积核，相当于全连接层


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.fc(x).squeeze(2).squeeze(2)
        return x


# # #VGG预训练模型
# class VGG(nn.Module):
#     def __init__(self, model_name):
#         super(VGG, self).__init__()
#         basemodel = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
#         basemodel = nn.Sequential(*list(basemodel.children())[:-7])
#         basemodel = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(7,7)),*list(basemodel.children())[1][:-2])
#         print(basemodel)
#         self.features = basemodel

#         if model_name == "vgg11":
#             num_ch = 512

#         self.fc = nn.Conv2d(num_ch, 50, 1)
#         self.pool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         x = self.features(x)
#         print(x.size())
#         x = self.pool(x)
#         x = self.fc(x).squeeze(2).squeeze(2)
#         return x








if __name__=='__main__':
    # resnet 模型测试 passed
    model = resnet("resnet18")
    model.eval()
    print(model)
    input = torch.randn(32,3,128,1723)
    y = model(input)
    print(y.size())
    # from torchsummary import summary
    # summary(model, (3, 128, 1723))
    pass
