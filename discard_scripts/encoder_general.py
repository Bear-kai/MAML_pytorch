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
            ('conv1', nn.Conv2d(num_channel, h_dim, kernel_size=3 ,padding=1)),  # return F.conv2d
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
            ('pool4', nn.MaxPool2d(2, ceil_mode=True))                           # F.max_pool2d
        ]))
        self.add_module('fc', nn.Linear(embed_dim, class_num))
        self.weights_init()

    def weights_init(self):
        ''' Set weights to Gaussian, biases to zero '''
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        #torch.cuda.manual_seed_all(1337)

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


    def weights_reset(self, fast_weights):
        '''
            2020.4.17验证失败, 该函数无法满足二阶需求，弃
        '''
        # 1. 因内存共享，fast_weights数值可自动传递到encoder的参数中，但.data后无法二阶梯度，可用于inner_update
        # for name, param in self.named_parameters():
        #     param.data = fast_weights[name].data                        

        # 2. 将fast_weights赋给encoder参数，重新定义了变量，同样无法二阶梯度，且optimizer.params中不会自动更新m.weight
        for name, m in self.named_modules():
            name_w = name + '.weight'
            name_b = name + '.bias'
            # 一般，有weight不一定有bias，有bias一定有weight，所以name_b放在后面判断
            if name_w not in fast_weights.keys():
                continue
            m.weight = nn.Parameter(fast_weights[name_w])
            if name_b not in fast_weights.keys():   
                continue
            m.bias = nn.Parameter(fast_weights[name_b])


    def forward_hooks(self, fast_weights):
        """ m: module, x: tuple, y: tensor 
            2020.4.20 失败, 忘了前向hook不能修改输入输出: The hook should not modify the input or output. 
            若Pytorch框架能支持动态改变forward的输入输出，则该实现方式对于不同的网络结构是普适的
        """
        handler_ls = []
        keys_ls = list(fast_weights.keys())

        def hook_fn(m, x, y): 
            global idx      # 须在forward()最后设置idx=0

            if isinstance(m, nn.Conv2d):
                y = F.conv2d(input=x[0], 
                            weight=fast_weights[keys_ls[idx]], 
                            bias=fast_weights[keys_ls[idx+1]],
                            stride=m.stride,
                            padding=m.padding,
                            dilation=m.dilation,
                            groups=m.groups )
                idx += 2
            
            elif isinstance(m, nn.BatchNorm2d):
                y = F.batch_norm(input=x[0], 
                                running_mean=m.running_mean, 
                                running_var=m.running_var,
                                weight=fast_weights[keys_ls[idx]],
                                bias=fast_weights[keys_ls[idx+1]],
                                training=m.training,
                                momentum=m.momentum,
                                eps=m.eps )
                idx += 2

            elif isinstance(m, nn.PReLU):
                y = F.prelu(input=x[0],
                            weight=fast_weights[keys_ls[idx]] )
                idx += 1

            elif isinstance(m, nn.Linear):
                y = F.linear(input=x[0], 
                            weight=fast_weights[keys_ls[idx]], 
                            bias=fast_weights[keys_ls[idx+1]] )
                idx += 2

        def add_hooks(m):
            if len(list(m.children())) > 0:     # skip modules that have children
                return
            if len(m._parameters) == 0:         # skip modules without learned parameters
                return
            print("Register hook for module %s" % str(m))
            handler = m.register_forward_hook(hook_fn)
            handler_ls.append(handler)

        self.apply(add_hooks)

        return handler_ls


    def forward(self, x):  
        out = self.fea(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        # idx = 0       #  used in forward_hooks

        return out
