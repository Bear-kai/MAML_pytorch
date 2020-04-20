import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import argparse
from torch.nn.modules.loss import CrossEntropyLoss
from encoder_general import CNNEncoder
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("-il","--inner_lr", type = float, default = 0.01)
parser.add_argument("-ml","--meta_lr", type = float, default = 0.001)
args = parser.parse_args()

INNER_LR = args.inner_lr
META_LR = args.meta_lr

CLASS_NUM = 10           # 10-way classification
BATCH_SIZE = 32
num_channel = 3
h_dim = 64
embed_dim = h_dim*4
epoch = 3
batch = 10
gpu_id = None # [1]      # 若在cpu上运行，则置为None


def set_seed(seed=666):
    torch.manual_seed(seed)             # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)        # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)    # 为所有GPU设置随机种子

set_seed()


def get_random_data(device=torch.device("cpu")):
    D1 = torch.randn(BATCH_SIZE,3,28,28).to(device)
    D1_label  = torch.randint(low=0,high=9,size=(BATCH_SIZE,)).to(device).long()

    return D1, D1_label


def main():
    """ 1. 下面是以单任务、单步更新为例, 基于pytorch内建函数, 实现最简洁的MAML一阶近似
        2. 该实现等价于在MAML_2nd.py中设置inner_update_num=1, batch_task_num=1, second_order=False
        3. 支持使用CPU和多GPU, CPU和单GPU结果一致,多GPU结果有差异，原因应该在BN，可尝试使用nn.SyncBatchNorm(本文未测试) 
    """
    # init neural network, optimizer, lr_scheduler, loss_function
    model = CNNEncoder(CLASS_NUM, num_channel, h_dim, embed_dim)
    # optimizer_1，optimizer_2和model.parameters中的参数共享内存，修改任一个，其余两个会自动改变
    optimizer_in = torch.optim.SGD(model.parameters(), lr=INNER_LR, momentum=0. )
    optimizer_out = torch.optim.SGD(model.parameters(), lr=META_LR, momentum=0.9 )  
    loss_fn = CrossEntropyLoss()

    # 支持使用CPU和多GPU
    if (gpu_id is not None) and torch.cuda.is_available():
        device = torch.device("cuda:%d"%gpu_id[0])
        if len(gpu_id)>1:       # multi-GPU setting
            model = nn.DataParallel(model,device_ids=gpu_id)
            model = model.to(device)
        elif len(gpu_id)==1:    # single-GPU setting
            model = model.to(device)
    else:
        device=torch.device("cpu")

    # state_dict()包含running_mean等所有参数; parameters()和named_parameters()仅包含需要优化的参数
    copy_weights = deepcopy(model.state_dict())     

    model.train()
    for i in range(epoch):
        for j in range(batch):
            print('epoch-%d, task-%d'%(i,j))
            
            # base learner (inner update) 
            inputs_1, labels_1 = get_random_data(device)
            pred_1  = model(inputs_1)
            loss_1 = loss_fn(pred_1, labels_1)
            optimizer_in.zero_grad()
            loss_1.backward()              
            optimizer_in.step()     

            # meta learner (outer update)
            inputs_2, labels_2 = get_random_data(device)
            pred_2 = model(inputs_2)
            loss_2 = loss_fn(pred_2, labels_2)
            # load_state_dict操作须在optimizer.step()之前
            model.load_state_dict(copy_weights)     
            optimizer_out.zero_grad()
            # 由于optimizer_in.step()中是基于.data更新模型参数，所以下面是求一阶梯度
            loss_2.backward()         
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_out.step()

            copy_weights = deepcopy(model.state_dict())


if __name__ == '__main__':
    main()

