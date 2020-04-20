### 概述
构建一个toy net，测试不参与运算的变量是否会更新&如何更新，加深对pytorch框架参数更新逻辑的理解。

### 起因
实现随机深度策略时，在block内部进行requires_grad=True/False操作会报错
（后面测试知道其实是DataParallel的锅）ref: [1](https://github.com/sciencefans/trojans-face-recognizer/blob/master/EfficientPolyFace/EfficientPolyFace.py), [2](https://github.com/shamangary/Pytorch-Stochastic-Depth-Resnet/blob/master/TYY_stodepth_lineardecay.py)

### 测试代码
结论见后

```python
# 以下代码中，需要设置或取消对应的代码屏蔽，完成不同的测试内容
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(20,20,3,1,1)
        self.bn = nn.BatchNorm2d(20)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20,20,3,1,1)
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4_1', nn.Conv2d(20, 20, 3, 1, 1)),
            ('conv4_2', nn.Conv2d(20, 20, 3, 1, 1))
        ]))		# 测试不同的block构建是否有影响
        self.conv5 = ConvBlock()
        self.conv6 = nn.Sequential(
            ConvBlock(),
            ConvBlock()
        )       # 测试不同的block构建是否有影响
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.5]))
        self.run = 0

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
        # if self.run:   # torch.equal(self.m.sample(), torch.ones(1)):
        #     print('run conv3')
        #     x = self.conv3(x)
        #     # self.run = 0      # 设置仅第一次运行时用
        # else:
        #     print('skip conv3')
        #     self.run += 1       # 设置第一次后运行时用
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # self.conv6._modules['0']._modules['conv'].weight.detach_()    
        x = self.conv6(x)

        # self.conv3.weight.requires_grad = False                                 # 可以直接在forward中设置requires_grad
        # self.conv4._modules['conv4_1'].weight.requires_grad = False             # 可以直接在forward中设置requires_grad
        # self.conv5._modules['conv'].weight.requires_grad = False                # 可以直接在forward中设置requires_grad
        # self.conv6._modules['0']._modules['conv'].weight.requires_grad = False  # 可以直接在forward中设置requires_grad

        # for item in self.conv6.modules():                                       # 可以直接在forward中设置requires_grad
        #     if isinstance(item, nn.Conv2d):
        #         item_no_grad = item.weight.detach()
        #         item.weight.requires_grad = False

        # self.conv6._modules['0']._modules['conv'].weight.detach_()
        # weight_no_grad = self.conv6._modules['0']._modules['conv'].weight.detach()

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x    # F.log_softmax(x, dim=1)

gpu_id = [3]
device = torch.device("cuda:%d"%gpu_id[0] if torch.cuda.is_available() else "cpu") #
model = Net()
if len(gpu_id)>1:  
   model = nn.DataParallel(model,device_ids=gpu_id)
   model = model.to(device)
elif len(gpu_id)==1:
   model = model.to(device)

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = polynet_stodepth()
# model = model.cuda()  

LOSS = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9 )
# del OPTIMIZER.param_groups[0]['params'][4]	                               # 移除优化器中的参数
# OPTIMIZER.param_groups[0]['params'].append(model._modules['conv3'].weight)   # 将参数加入优化器参数列表

epoch = 10
batch = 50
bs = 2

model.train()
# model.conv3.weight.requires_grad = False
for i in range(epoch):
    print('===== epoch %d'%i)
    for j in range(batch):
        print('epoch-%d, batch-%d'%(i,j))
        inputs = torch.randn(bs,1,28,28).to(device)
        labels = torch.randint(low=0,high=9,size=(bs,)).to(device)
        # inputs = torch.randn(bs, 3, 235, 235).cuda()  #.to(device)
        # labels = torch.randint(low=0, high=9, size=(bs,)).cuda()  #.to(device)

        pred = model(inputs)
        loss = LOSS(pred, labels)

        # for m in model.modules():
        #     print(str(m.__class__))     # 打印的是各操作的类型，如Conv2d，类型是有重复的

        # for pname, p in model.named_parameters():
        #     print(pname)                # 打印的是各操作的名字，名字是唯一标识的
        
        OPTIMIZER.zero_grad()
        loss.backward()
        # model.conv3.weight.requires_grad = False
        OPTIMIZER.step()
print('done')
```

### 实验观察结论
 1. 初始化各模块如self.conv3后，其_grad值为None
 2. self.conv3只有在forward中运行了，才会被纳入graph中，才会在loss.backward时计算_grad
 3. 若只有第一次forward时self.conv3进行了运算，其他都跳过，则后续self.conv3的_grad为0.0
**——>** 原因：会被zero_grad；注意梯度为0，而非为None，self.conv3的参数还是会被更新，这是由于实际优化时有动量的存在，梯度为0时也会使参数值变化
 4. 若第一次forward时跳过了self.conv3，后面又加入？
**——>**  第一次为None,后面正常计算梯度
 5. 若_grad一直为None，是否会更新参数呢？--> 不会更新
 6. 参与forward计算，但在model外部，一开始就设置self.conv3的requires_grad为False，参数是否更新？
**——>**  不会计算_grad，即一直为None,故不更新参数
 7. 参与forward计算，但在model外部，在backward和step之间才设置self.conv3的requires_grad为False，参数是否更新？
**——>**  会更新！因为backward后已经有了_grad生成，则只要参数收录于optimizer，就会被更新；
**——>**  同样，下一次迭代时，由于zero_grad，_grad会为[0.0,...]，然后由于requires_grad为False，所以不会计算新的_grad
**——>**  参数也是基于动量而改变的
 8. 参与forward计算，但forward内部，设置self.conv3的requires_grad为False，参数是否更新？
**——>**  经测试，竟然可以直接在forward内部赋值；经分析，可转为case 6
 9. 参与forward计算，设置requires_grad=True，但是在optimizer中去掉self.conv3，参数是否更新？
**——>**  不会更新，但是_grad依然会计算，但不会被置零，（看来zero_grad是按照optimizer中的参数列表进行操作,想来也合理，本来就是optimizer.zero_grad()）
**——>**  补充：不参与forward计算，从optimizer中删除参数，保留state中参数-动量dict，则参数不更新，梯度和动量不计算且保持原值不变；sgd.py的step中：buf.mul_(momentum).add_(1 - dampening, d_p), d_p是当次的梯度，buf是动量，buf乘以系数momentum再加上(1-dampending)*d_p作为新的梯度用于更新参数
**——>**  补充：参数更新对应到optimizer中就是的param_groups中的参数值改变，state的dict{参数i:动量i}的参数i自动相应改变，动量更新如9中所述。
 10. 分析测试是什么造成PolyFaceNet中的forward内部不能设置requires_grad ？
**——>**  经测试，各种情况下，forward内部都能直接设置requires_grad
 11. 测试DataParallel
**——>**  *****经测试，就是这个导致的RuntimeError [[issue-3] ](https://github.com/sciencefans/trojans-face-recognizer/issues/3)*****
实际上进入model(x)之前，model的参数都是leaf变量(requires_grad=True，可手动设置)，进入model.forward()后，就变成非leaf变量了(requires_grad=True,不能手动设置修改)，退出forward()后，恢复成leaf变量（requires_grad=True,可手动设置修改）
**——>**  在forward内部，在conv6(x)的前或后，使用self.conv6...weight0.detach_()，可以将weight0设置为leaf变量，且grad_fn=None, requires_grad=False，均不会对self.conv6中后续的其他weight1有影响，但是退出forward后，weight0还是会还原requires_grad=True...
**——>**  经测试，在PolyFaceNet中设置self.keep_block，若不对model用DataParallel，则可正常改变model的keep_block属性。若对model用DataParallel，则model保留__init__中初始化的keep_block不变，或者未初始化时，无此属性（即不会因为forward添加）
**——>**  被DataParallel包装过的model,会多一个前缀'modules'，如PolyFaceNet.py中: model.module.blockA_multi.0  （单gpu时是model.blockA_multi.0）
**——>** 联想保存多gpu训练的模型时，要调用model.module.state_dict(),单gpu时是model.state_dict()
**——>** 更多：DataParallel也是nn.Module, 传入的model保存在DataParallel的self.module变量里面,所以为什么多了一个前缀
**——>** 回过头看，因为DataParallel是将模型复制到各gpu上分别进行前向传播，若允许在前向中修改属性，会引起歧义：比如gpu1上将blockA保留了，但是gpu2上又将其删除，则最终综合结果时，主gpu将"懵逼"，所以逻辑上会禁止在前向中修改，这么一想，合情合理。
 12. 测试os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'实现多GPU训练
**——>**  .cuda()和.to(device)均失败！模型和数据只会被移到默认的gpu0上（10873/11178M, 353M/11178M）
 13. DataParallel是单机多卡，对于多机多卡：https://zhuanlan.zhihu.com/p/68717029

### 总结
正常情况下，不考虑上述测试中的各种特殊情形

 1. **requires_grad管梯度计算**，参数的该属性为True时就会计算该参数的梯度
 2. **optimizer.param_groups管参数更新**，若参数w被收录于param_groups中，参数w就会参与“**更新过程**”，否则不参与。（注意，**这里参数w参与更新过程是指optimizer.step()中会遍历到w**，但w数值可能不改变，比如requires_grad=False导致grad=None，将会跳过w： ‘ if p.grad is None: continue ’）
 3. **对于平时采用预训练模型，比如要固定backbone训练head**，则有两种方案实现：a. 定义optimizer时不收录backbone的params，无所谓requires_grad的取值；b. **设置backbone的params的requires_grad=False**，无所谓optimizer中收录不收录。（通过上述分析，自然是采用方案b为佳，因为backward时不用再计算backbone参数的梯度！大大加快训练速度且节省大量显存）
 4. 结合pytorch的动态图机制理解（[点这里](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)），**前向负责构图**，**反向负责求梯度**（对所有requires_grad=True的变量算梯度），**反向完毕则释放图**，因此若需要二次反向，需要保证图没有被释放掉：`retain_graph=True`。
