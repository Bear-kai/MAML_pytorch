import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import argparse
from torch.nn.modules.loss import CrossEntropyLoss
from encoder_general import CNNEncoder
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("-il","--inner_lr", type = float, default = 0.1)
parser.add_argument("-ml","--meta_lr", type = float, default = 0.01)
parser.add_argument("-i","--inner_update_num", type = float, default = 3)
parser.add_argument("-m","--meta_train_iter",type = int, default= 1000)
args = parser.parse_args()

INNER_LR = args.inner_lr
META_LR = args.meta_lr
INNER_UPDATE_NUM = args.inner_update_num
META_TRAIN_ITER = args.meta_train_iter

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


def set_meta_gd(model,weights_before,weights_after):
    for name, param in model.named_parameters():
        # optimizer.step()中param-lr*grad带符号, 因此下面是before-after
        param.grad = weights_before[name] - weights_after[name]
    return model


def main():
    """ 
    """
    # init neural network, optimizer, lr_scheduler, loss_function
    model = CNNEncoder(CLASS_NUM, num_channel, h_dim, embed_dim)
    # optimizer_1，optimizer_2和model.parameters中的参数共享内存，修改任一个，其余两个会自动改变
    optimizer_in = torch.optim.SGD(model.parameters(), lr=INNER_LR, momentum=0. )
    optimizer_out = torch.optim.SGD(model.parameters(), lr=META_LR, momentum=0. )  
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

    model.train()
    # MAML的一个task包含Dtr和Dte, reptile的一个task只有Dtr（即没有了inputs_2,labels_2）
    for i in range(META_TRAIN_ITER):  
        weights_before = deepcopy(model.state_dict())   
        
        # base learner (inner update) 
        inputs_1, labels_1 = get_random_data(device)    
        for j in range(INNER_UPDATE_NUM):
            pred_1  = model(inputs_1)
            loss_1 = loss_fn(pred_1, labels_1)
            optimizer_in.zero_grad()
            loss_1.backward()              
            optimizer_in.step()    

        # meta learner (outer update)
        weights_after = deepcopy(model.state_dict())
        model.load_state_dict(weights_before)  
        model = set_meta_gd(model,weights_before,weights_after)     
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_out.step()


if __name__ == '__main__':
    main()

