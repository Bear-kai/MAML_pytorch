import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from collections import OrderedDict


class CNNEncoder(nn.Module):
    def __init__(self ,class_num, num_channel ,h_dim ,embed_dim):
        super(CNNEncoder ,self).__init__()

        self.fea = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(num_channel, h_dim, kernel_size=3 ,padding=1)),  
            ('bn1', nn.BatchNorm2d(h_dim, momentum=1, affine=True)),
            ('relu1', nn.ReLU()),   # inplace=True
            ('pool1', nn.MaxPool2d(2, ceil_mode=True)),  # 2,2
            ('conv2', nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(h_dim, momentum=1, affine=True)),
            ('relu2', nn.ReLU()),  # inplace=True
            ('pool2', nn.MaxPool2d(2, ceil_mode=True)),
            ('conv3', nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(h_dim, momentum=1, affine=True)),
            ('relu3', nn.ReLU()),  # inplace=True
            ('pool3', nn.MaxPool2d(2, ceil_mode=True)),
            ('conv4', nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(h_dim, momentum=1, affine=True)),
            ('relu4', nn.ReLU()),  # inplace=True
            ('pool4', nn.MaxPool2d(2, ceil_mode=True))                          
        ]))
        self.add_module('fc', nn.Linear(embed_dim, class_num))
        self.weights_init()

    def weights_init(self):
        ''' Set weights to Gaussian, biases to zero '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_() + 1
                m.bias.data = torch.ones(m.bias.data.size())

    def batch_norm(self, input, weight=None, bias=None,training=True, eps=1e-5, momentum=0.1):
        # This hack only works when momentum is 1 and avoids needing to track running stats
        running_mean = torch.zeros(np.prod(np.array(input.data.size()[1])))
        running_var = torch.ones(np.prod(np.array(input.data.size()[1])))

        return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

    def forward(self, x, weights=None):  # nn.Module要求override的函数
        if weights == None:
            out = self.fea(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:
            out = F.conv2d(x, weights['fea.conv1.weight'], weights['fea.conv1.bias'],padding=1)
            out = self.batch_norm(out, weights['fea.bn1.weight'], weights['fea.bn1.bias'])  # ,momentum=1
            out = F.threshold(out, 0, 0)   #  threshold, value. 当输入x>threshold时，输出为x，当x<threshold,输出为value
            out = F.max_pool2d(out, kernel_size=2, ceil_mode=True)   # stride=2,

            out = F.conv2d(out, weights['fea.conv2.weight'], weights['fea.conv2.bias'],padding=1)
            out = self.batch_norm(out, weights['fea.bn2.weight'], weights['fea.bn2.bias']) # , momentum=1
            out = F.threshold(out, 0, 0)
            out = F.max_pool2d(out, kernel_size=2, ceil_mode=True)

            out = F.conv2d(out, weights['fea.conv3.weight'], weights['fea.conv3.bias'],padding=1)
            out = self.batch_norm(out, weights['fea.bn3.weight'], weights['fea.bn3.bias']) # , momentum=1
            out = F.threshold(out, 0, 0)
            out = F.max_pool2d(out, kernel_size=2, ceil_mode=True)

            out = F.conv2d(out, weights['fea.conv4.weight'], weights['fea.conv4.bias'],padding=1)
            out = self.batch_norm(out, weights['fea.bn4.weight'], weights['fea.bn4.bias']) # , momentum=1
            out = F.threshold(out, 0, 0)
            out = F.max_pool2d(out, kernel_size=2, ceil_mode=True)

            out = out.view(out.size(0), -1)
            out = F.linear(out, weights['fc.weight'], weights['fc.bias'])

        return out

    