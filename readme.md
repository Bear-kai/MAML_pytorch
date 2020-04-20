## MAML-Pytorch
This is a PyTorch implementation of [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400).

For Official Tensorflow Implementation, please visit [Here](https://github.com/cbfinn/maml).

## requirements
1. python: 3.x
2. pytorch 1.0+

## files & dirs
1. encoder.py

    模型定义文件，需要手动定义指定参数的前向传播函数（见`forward()`函数），更换模型时这个过程得重新进行一遍，代码迁移性差

2. encoder_general.py

    相对普适的模型定义文件，迁移性稍好，更换模型时不用从零开始定义指定参数的前向传播函数，目前仅适于VGG类串行网络结构

3. **MAML_1st.py**

    简洁的MAML一阶近似版实现（基于pytorch内建函数）

4. MAML_2nd.py

    相对普适的MAML二阶版实现（基于手动更新网络参数）; 通过设置参数，可退化为`MAML_1st.py`中的一阶版近似;

    目前github上的第三方开源实现，有的并未实现二阶梯度更新，有的实现繁琐、代码迁移性差，主要表现在基于指定参数的前向传播这部分（对应inner_update中的`encoder.forward_net(x, fast_weights)`）

    本文最初目的是实现普适版的MAML二阶更新代码，普适是指对于任意模型结构，在不用修改原模型定义文件，或者仅需要添加一个类函数的前提下，即可方便地将MAML二阶更新代码用于该模型;

    经过若干尝试，目前仅实现对于VGG类串行网络结构的普适代码，无法直接迁移到ResNet这类含分支的网络结构上，问题的根源在于上述提到的inner_update中的指定参数前向传播。（以个人目前对pytorch框架的理解，可能无法实现这种普适的代码，欢迎讨论交流`xiongkai4925@cvte.com` / `bearkai1992@qq.com`）

5. test_grad_20200408.py

    编写的一些tiny_test，帮助理解pytorch的`backward()`和`grad()`, `.data`和.`detach()`, `retain_graph`和`create_graph`等概念，帮助验证MAML一阶/二阶梯度更新中的一些操作是否符合预期。

    若只需要实现MAML梯度更新，可以忽略该文件。

6. test_param_20191016.py

    编写的一些tiny_test，帮助理解pytorch的参数更新机制，并排查实现深度随机策略时的bug。

    若只需要实现MAML梯度更新，可以忽略该文件。对pytorch框架感兴趣的同学，建议阅读文件最后的总结内容。

7. discard_scripts

    一些失败的尝试，涉及`hook_function`, `name_modules()`等概念；若只需要实现MAML梯度更新，可以忽略该文件。

## Note
1. 上述努力实现普适版的MAML二阶代码，更多是为了研究上的完备性，并加深对深度框架的理解。实际上，对于小模型，参考`encoder.py`重新定义模型的`forward()`，工作量也还好，对于大模型，实际用不到二阶梯度更新，因为太占内存，速度太慢。因此，重点关注**MAML_1st.py**。
2. To do : Reptile and training tricks of MAML.

