import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, OrderedDict
from copy import deepcopy


def tiny_test_1():
    """
    测试retain_graph和create_graph
    """
    # ========== 测试retain_graph, 以下两种等价
    # x = torch.tensor([1.,2.],requires_grad=True)
    # y = x.pow(2) + 1
    # z = 0.5*y
    # loss_1 = z.mean()
    # loss_2 = z.sum()
    # sum_loss = loss_1 + loss_2
    # sum_loss.backward()
    # print(x.grad)
    
    # a = torch.tensor([1.,2.],requires_grad=True)
    # b = a.pow(2) + 1
    # c = 0.5*b
    # loss_3 = c.mean()
    # loss_4 = c.sum()
    # loss_3.backward(retain_graph=True)
    # print(a.grad)
    # loss_4.backward()
    # print(a.grad)

    # ==========
    # # retain_graph的应用，可参考GAN训练，一次前向，两个loss反向 (https://zhuanlan.zhihu.com/p/43843694)，伪代码如下：
    # # 前向
    # gen_imgs = generator(z)                     # 从噪声中生成假数据    
    # pred_gen = discriminator(gen_imgs)          # 判别器对假数据的输出  
    # pred_real = discriminator(real_imgs)        # 判别器对真数据的输出  
    # # 反向--更新判别器
    # d_loss = 0.5*(adversarial_loss(pred_real, valid) + adversarial_loss(pred_gen, fake))
    # optimizer_D.zero_grad()                     
    # d_loss.backward(retain_graph=True)          
    # optimizer_D.step()
    # # 反向--更新生成器
    # g_loss = adversarial_loss(pred_gen, valid)  
    # optimizer_G.zero_grad()                     
    # g_loss.backward() 
    # optimizer_G.step()

    # ========== 不涉及高阶微分时，效果上create_graph等同于retain_graph (因为开启create_graph会默认开启retain_graph)
    # # 此例中设置retain_graph或create_graph, 两次backward()后x.grad均为[3.,6.], 
    # # 分析：sum_loss的反向，并不涉及高阶微分，多次backward()仅是copy梯度并在叶节点上累加
    # x = torch.tensor([1.,2.],requires_grad=True)
    # y = x.pow(2) + 1
    # z = 0.5*y
    # loss_1 = z.mean()
    # loss_2 = z.sum()
    # sum_loss = loss_1 + loss_2
    # sum_loss.backward(create_graph=True)    # retain_graph=True
    # print(x.grad)
    # x.grad = None
    # sum_loss.backward()
    # print(x.grad)

    # ========== 测试一二三阶梯度
    x = torch.tensor([1.,2.],requires_grad=True)
    y = x.pow(2) + 1
    z = 0.5*y
    loss_1 = z.mean()
    loss_2 = z.sum()
    sum_loss = loss_1 + loss_2

    # 一阶微分，结果1.5x=[1.5,3.]
    sum_loss.backward(create_graph=True)    
    print(sum_loss, x.grad)   

    # 二阶微分，结果2*[1.5,1.5]=[3.,3.]
    # 分析：Ld2/x1 = Ld2/x1' * x1'/x1 = 2 * 1.5 = 3.
    # 若上面sum_loss.backward未设置create_graph，或仅设置retain_graph，则Ld2反向时会报错：
    # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    # 因此也可反向验证，若retain_graph=True后可进行二次backward，说明不涉及二阶微分
    Ld2 = 2*x.grad.sum()
    x.grad = None
    Ld2.backward(retain_graph=True)
    print(Ld2, x.grad)   

    # 三阶微分，结果0.5*[0.,0.]=[0.,0.]
    # 分析：Ld3/x1 = Ld3/x1'' * x1''/x1 = 0.5 * 0. = 0.
    # 若上面Ld2.backward未设置create_graph，或仅设置retain_graph，则Ld3反向时会报错
    Ld3 = x.grad.mean()
    x.grad = None
    Ld3.backward()
    print(Ld3, x.grad)


def tiny_test_2():
    """
    测试修改变量取值，是否影响梯度计算，对照tiny_test_4和tiny_test_5
    """
    # ========== 模拟一阶微分中的权值恢复 model.load_state_dict(copy_weights)
    a = torch.tensor([1.,2.], requires_grad=True) 
    b = a.pow(2) + 1
    c = b.sum()
    # (1) 新生成变量a，会影响最终结果：a.grad=None
    # a = torch.tensor([2.,3.], requires_grad=True)   
    # (2) 仅修改变量a的取值，不影响最终结果：a.grad=[2.,4.]
    # a.data = torch.tensor([2.,3.])      
    # (3) 同时修改变量a的取值和grad，最终结果是两grad累加：a.grad=[2.5,4.6]，因此一般需要先zero_grad
    a.data = torch.tensor([2.,3.])   
    a.grad = torch.tensor([0.5,0.6])
    c.backward()
    print(a.grad)

    # ========== 模拟MAML二阶微分中的权值恢复theta'-->theta：model.load_state_dict(copy_weights)
    x = torch.tensor([1.,2.], requires_grad=True)
    y = x.pow(2) + 1
    z = y.sum()
    z.backward(create_graph=True)
    print(x.grad)                         # x.grad=[2.,4.]
    # (1) 常规是变量参与运算
    L = 0.5*x.grad.pow(2) + x.grad + 1    
    # (2) 若只用.data参与运算，backward时报错，因L本身已不需要计算梯度
    # L = 0.5*x.grad.data.pow(2) + x.grad.data + 1    
    # x.data = torch.tensor([1.3, 3.6])   # 改变x取值不影响x.grad结果 (引申：MAML二阶梯度正确，只用在step()更新参数前，load保存的参数theta)
    x.grad = None                         # zero_grad或置为None，则只涉及二阶微分的结果，没有累加一阶微分
    L.sum().backward()                    
    # x.data = torch.tensor([1.4, 3.7])   # 改变x取值不影响x.grad结果
    print(x.grad)                         # x.grad = tensor([ 6., 10.]); 分析：x'=2*x=[2.,4.], L/x = L/x' * x'/x = (x'+1)*2

    # ========== 模拟optimizer.step中的参数更新方式，修改变量取值
    # 更新x是基于.data，因此L的计算并不涉及x'，
    # 不设置create_graph时，相当于构建了两个图，分别计算z和L，只不过共享leaf x; 设置create_graph时，相当于在原图上，增加了跟L相关的节点
    x = torch.tensor([1.,2.], requires_grad=True)
    y = x.pow(2) + 1
    z = y.sum()
    z.backward(create_graph=True)                         # 
    print(x.grad)                        # tensor([2., 4.], grad_fn=<CloneBackward>)
    x.data = x.data - 0.1 * x.grad.data  # [0.8,1.6] 
    L = 2*x + 1                          # [2.6,4.2]
    x.grad = None
    L.sum().backward() 
    print(x.grad)                        # tensor([2., 2.])，对应L=2x+1中的系数2

    x = torch.tensor([1.,2.], requires_grad=True)
    y = x.pow(2) + 1
    z = y.sum()
    z.backward(create_graph=True)                         # 
    print(x.grad)                        # tensor([2., 4.], grad_fn=<CloneBackward>)
    x1 = x - 0.1 * x.grad                # 因x1不是leaf，故L反传后x1.grad=None
    L = 2*x1 + 1                         # [2.6,4.2]
    x.grad = None
    L.sum().backward() 
    print(x.grad)                        # tensor([1.6, 1.6])，这里才真正涉及二阶微分

    x = torch.tensor([1.,2.], requires_grad=True)
    y = x.pow(2) + 1
    z = y.sum()
    z.backward(create_graph=True)        
    print(x.grad)                        # tensor([2., 4.], grad_fn=<CloneBackward>)
    x = x - 0.1 * x.grad                 # 构建graph时不能采用同名变量! 执行完该句后，x.grad=None, x.grad_fn为SubBackward0, x.is_leaf=False
    L = 2*x + 1                          # 对于网络参数值的更新，因为变量名是要保持不变的，为了保持is_leaf，所以optimizer.step中是基于.data计算
    # x.grad = torch.tensor([0.4,0.4])   
    L.sum().backward() 
    print(x.grad)                        # 因x不是leaf，故L反传后x.grad=None
 

def tiny_test_3():
    """
    测试tensor.detach()和tensor.detach_()； autograd.backward()和autograd.grad()
    """
    # 若不考虑detach，因x.requires_grad=True, y和z也自动requires_grad, 但只有x是leaf node，所以只有x的梯度会保留，y的梯度backward完毕即被释放
    x = torch.tensor([1.,2.], requires_grad=True)
    y = x.pow(2) + 1
    z_1 = y.sum()                         # 计算z_1后先做y.detach_()，再.grad(z_1,y)报错，因y无需求梯度了；.grad(z_1,x)或z_1.backward正常运行
    q = y.detach()                        # --> q.requires_grad=False, y不变
    y.detach_()                           # --> y.requires_grad=False
    z = y.sum() + q.sum()                 # 因y和q均不要求梯度，因此z.requires_grad=False，对z做backward或grad会报错
    z_2 = z_1 + y.sum()                   # 构建z_1时还没有detach y,所以z_1可正常反传, 又有z_2反传等同于z_1反传

    # (1) torch.autograd.backward: Computes the sum of gradients of given tensors w.r.t. graph leaves.
    z_2.backward()                     
    # (2) torch.autograd.grad: Computes and returns the sum of gradients of outputs w.r.t. the inputs.
    # xy_grad = torch.autograd.grad(z_1,[x,y])    # return: tuple  也包括create_graph和retain_graph参数
    # print(xy_grad)                      # (tensor([2., 4.], grad_fn=<MulBackward0>), tensor([1., 1.]))
    print(y.grad)                         # y.grad = None
    print(x.grad)                         # x.grad = tensor([2., 4.], grad_fn=<CloneBackward>), 

    # 注意：backward()和grad()计算的x.grad的grad_fn不同，且grad()计算的梯度只作为返回值，不会自动累加到x.grad中
    # x.grad = xy_grad[0]


def tiny_test_4():
    """
    对detach后的变量进行修改，若变量取值涉及梯度计算，则backward会报错，若不涉及，则不报错(且能保证梯度计算正确)
    detach后的变量, requires_grad=False，且仍会被autograd追踪
    """
    a = torch.tensor([1, 2, 3.], requires_grad=True)
    print(a.grad)
    out = a.sigmoid()
    print(out)              # tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
    c = out.detach()        # c和out指向同一内存地址
    print(c)                # tensor([0.7311, 0.8808, 0.9526])
    # c.zero_()               # 改变c会同步改变out
    # c.data[0] = 1.5       # 改变c会同步改变out
    # c.data = torch.tensor([1.5,1.6,1.7])    # 改变c不会影响到out, 相当于让c指向了新的地址
    # c = torch.tensor([2.5,2.6,2.7])         # 改变c不会影响到out, 相当于让c指向了新的地址
    # print(c)
    # print(out)
    out.sum().backward()    # 因为out中的元素取值涉及到梯度计算，因此backwar会报错:
    print(a.grad)           # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation

    a1 = torch.tensor([1, 2, 3.], requires_grad=True)
    out1 = a1.sigmoid().sum()
    print(out1)
    c1 = out1.detach()      # c1和out1指向同一内存地址
    print(c1)    
    c1.zero_()              # 改变c1会同步改变out1
    print(c1)
    print(out1)
    out1.backward()         # 注意对比：这里out已求和，其取值不涉及梯度计算，因此backwar正常计算
    print(a1.grad)          # tensor([0.1966, 0.1050, 0.0452])

    x = torch.tensor([1.,2.], requires_grad=True)
    y = x.pow(2) + 1
    q = y.detach()          # q和y指向同一内存地址                  
    z = y.sum() + q.sum()                 
    q[0] = 3                # 改变q会同步改变y             
    z.backward()            # 因为y的取值并不涉及梯度计算(y对x求导为2*x)，因此backward可正常运行
    print(y.grad)           # y.grad = None
    print(x.grad)           # x.grad = tensor([2., 4.])，这里x.grad的传递路径是z-->y-->x


def tiny_test_5():
    """
    对.data后的变量进行修改, backward均不会报错(但梯度计算可能错误)，对照tiny_test_4
    .data后的变量, requires_grad=False, 且不会被autograd追踪
    (因此，只要基于非.data构建计算图，并能进行backward，说明未对涉及梯度计算的值进行改动，则计算的梯度一定正确)
    """
    a = torch.tensor([1, 2, 3.], requires_grad=True)
    print(a.grad)
    out = a.sigmoid()
    print(out)
    c = out.data            # c和out指向同一内存地址
    print(c)    
    c.zero_()               # 改变c会同步改变out
    print(c)
    print(out)
    out.sum().backward()    # .data的修改不会被autograd追踪, 因此backward不会报错，但返回错误的梯度值
    print(a.grad)           # tensor([0., 0., 0.])


def tiny_test_6():
    """ 
    1. 变量detach之后参与的运算结果(这部分计算构建了新的graph)，计算梯度时不会传递到旧graph中的相关变量 (见位置1)
    2. graph构建完成后，将其中的某个变量detach，不会影响梯度在graph中的传递 (见位置2)
    """
    a = torch.tensor(1., requires_grad=True)
    b = torch.tensor(2., requires_grad=True)
    c = torch.tensor(3., requires_grad=True)
    x = a + b       # 1+2=3
    y = 2 * c       # 2*3=6
    # y.detach_()   # ***** 位置1：输出 6., 6., None, 即不会对c求梯度
    z = x * y       # 若改为z = x + y, 结果变为1.,1.,2.或1.,1.,None
    y.detach_()     # ***** 位置2：输出 6., 6., 6.,   即会对c求梯度
    z.backward()
    print(a.grad)   # 6.
    print(b.grad)   # 6.
    print(c.grad)   # 6.


if __name__ == "__main__":
    # tiny_test_1()
    # tiny_test_2()
    # tiny_test_3()
    # tiny_test_4()
    # tiny_test_5()
    tiny_test_6()

