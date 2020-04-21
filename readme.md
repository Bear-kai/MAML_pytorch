## MAML-Pytorch
This is a PyTorch implementation of [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400).

For Official Tensorflow Implementation, please visit [Here](https://github.com/cbfinn/maml).

## requirements
1. python: 3.x
2. pytorch 1.0+

## files & dirs
1. encoder.py

    模型定义文件，需要手动定义支持参数传递的`forward()`函数，更换模型时这个过程得重新进行一遍，代码迁移性差

2. encoder_general.py

    相对普适的模型定义文件，迁移性稍好，更换模型时不用从零开始定义支持参数传递的`forward()`函数，目前仅适于VGG类串行网络结构

3. **MAML_1st.py**

    简洁的MAML一阶近似版实现（基于pytorch内建函数）

4. MAML_2nd.py

    相对普适的MAML二阶版实现（基于手动更新网络参数）; 通过设置参数，可退化为`MAML_1st.py`中的一阶版近似;

    本文最初目的是实现普适版的MAML二阶更新代码，普适是指对于任意模型结构，在不用修改原模型定义文件，或者仅需要添加一个类函数的前提下，即可方便地将MAML二阶更新用于该模型;

    经过若干尝试，目前仅实现对于VGG类串行网络结构的普适代码，无法直接迁移到ResNet这类含分支的网络结构上，对ResNet需要重构其build block，使支持带参数的前向传播 （pytorch框架可能不支持本文对模型普适的需求，欢迎讨论交流`xiongkai4925@cvte.com` / `bearkai1992@qq.com`）

5. test_grad_20200408.py

    编写的一些tiny_test，帮助理解pytorch的`backward()`和`grad()`, `.data`和.`detach()`, `retain_graph`和`create_graph`等概念，帮助验证MAML一阶/二阶梯度更新中的一些操作是否符合预期。

    若只需要实现MAML梯度更新，可以忽略该文件。

6. test_param_20191016.md

    编写的一些tiny_test，帮助理解pytorch的参数更新机制，并排查实现深度随机策略时的bug。

    若只需要实现MAML梯度更新，可以忽略该文件。对pytorch框架感兴趣的同学，建议阅读文件最后的总结内容。

7. discard_scripts

    一些失败的尝试，涉及`hook_function`, `name_modules()`等概念；若只需要实现MAML梯度更新，可以忽略该文件。

## Note
1. 上述实现普适版MAML二阶，更多是为了研究上的完备性，顺便加深对深度框架的理解；
2. 实际上，对于小模型，参考`encoder.py`重新定义模型的`forward()`，工作量较小，对于大模型，比如`ResNet50`，需要重构其build block，使能进行指定参数的前向传播，此外大模型可能不太用到二阶梯度更新，因为Hessian矩阵太占内存，计算速度太慢。


## To do 
Reptile [[1](https://arxiv.org/abs/1803.02999)[, 2](https://openai.com/blog/reptile/#jump)]  and training tricks of MAML [[3](https://arxiv.org/abs/1810.09502)]

