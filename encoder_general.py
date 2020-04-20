import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from collections import OrderedDict

# global variable
idx = 0
input_x = None

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


    def forward(self, x):  
        out = self.fea(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


    def forward_net(self,x,fast_weights):
        """ 2020.4.20
            1. 可将此函数迁移添加至自己的模型定义文件中，并设置两个global变量: idx和input_x (见文件顶部)
            2. 在该函数中补充自己模型里的所有operation，比如添加nn.ConvTranspose2d (对应F.conv_transpose2d )
            3. 该实现仅支持VGG类的串行网络结构, 无法普适到ResNet等含分支的网络结构
        """
        global idx, input_x
        keys_ls = list(fast_weights.keys())
        if input_x is None:
            input_x = x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                input_x = F.conv2d(input=input_x, 
                                weight=fast_weights[keys_ls[idx]], 
                                bias=fast_weights[keys_ls[idx+1]],
                                stride=m.stride,
                                padding=m.padding,
                                dilation=m.dilation,
                                groups=m.groups )
                idx += 2
            
            elif isinstance(m, nn.BatchNorm2d):
                input_x = F.batch_norm(input=input_x, 
                                running_mean=m.running_mean, 
                                running_var=m.running_var,
                                weight=fast_weights[keys_ls[idx]],
                                bias=fast_weights[keys_ls[idx+1]],
                                training=m.training,
                                momentum=m.momentum,
                                eps=m.eps )
                idx += 2

            elif isinstance(m, nn.PReLU):
                input_x = F.prelu(input=input_x,
                                    weight=fast_weights[keys_ls[idx]] )
                idx += 1

            elif isinstance(m, nn.Linear):
                input_x = input_x.view(input_x.size(0), -1)     
                input_x = F.linear(input=input_x, 
                                    weight=fast_weights[keys_ls[idx]], 
                                    bias=fast_weights[keys_ls[idx+1]] )
                idx += 2

            # 以下是不含learned parameters的operations
            elif isinstance(m, nn.ReLU):
                input_x = F.relu( input=input_x, 
                                    inplace=m.inplace)

            elif isinstance(m, nn.MaxPool2d):
                input_x = F.max_pool2d(input=input_x,
                                        kernel_size=m.kernel_size, 
                                        stride=m.stride, 
                                        padding=m.padding, 
                                        dilation=m.dilation,
                                        ceil_mode=m.ceil_mode, 
                                        return_indices=m.return_indices)
        # clone()会开辟新的内存空间，且记录于computation graph中，区别于detach()
        out = input_x.clone()
        # reset global variable
        idx = 0
        input_x = None
        
        return out

