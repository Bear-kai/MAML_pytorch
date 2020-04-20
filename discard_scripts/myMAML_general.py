import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from torch.nn.modules.loss import CrossEntropyLoss
from encoder_general import CNNEncoder
from collections import OrderedDict
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("-m","--meta_train_iter",type = int, default= 1000)
parser.add_argument("-i","--inner_update_num", type = float, default = 1)
parser.add_argument("-b","--batch_task_num", type = float, default = 1)   # 每次iteration中采样的task数目
parser.add_argument("-il","--inner_lr", type = float, default = 0.01)
parser.add_argument("-ml","--meta_lr", type = float, default = 0.001)
args = parser.parse_args()

META_TRAIN_ITER = args.meta_train_iter
INNER_UPDATE_NUM = args.inner_update_num
BATCH_TASK_NUM = args.batch_task_num
INNER_LR = args.inner_lr
META_LR = args.meta_lr

CLASS_NUM = 10      # 10-way classification
BATCH_SIZE = 32
num_channel = 3
h_dim = 64
embed_dim = h_dim*4

gpu_id = [1]      # 若在cpu上运行，则置为None


def set_seed(seed=666):
    torch.manual_seed(seed)             # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)        # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)    # 为所有GPU设置随机种子

set_seed()


def get_random_data(device=torch.device("cpu")):
    D1 = torch.randn(BATCH_SIZE,3,28,28).to(device)
    D1_label  = torch.randint(low=0,high=9,size=(BATCH_SIZE,)).to(device).long()
    D2 = torch.randn(BATCH_SIZE,3,28,28).to(device)
    D2_label  = torch.randint(low=0,high=9,size=(BATCH_SIZE,)).to(device).long()

    return D1, D1_label, D2, D2_label


def remove_handler(handler_ls):
    for handler in handler_ls:
        handler_ls.remove()


def inner_update_1(encoder,loss_fn,device,second_order):
    encoder_copy = deepcopy(encoder)
    # get task data: D1 for train, D2 for 'test'
    D1, D1_label, D2, D2_label = get_random_data(device)
    # get model parameters, then manually update
    fast_weights = OrderedDict((name, param) for (name, param) in encoder.named_parameters())  
    
    for i in range(INNER_UPDATE_NUM):
        if i == 0:
            pred = encoder_copy(D1)  
            loss1 = loss_fn(pred, D1_label)               
            grads = torch.autograd.grad(loss1, encoder_copy.parameters(), create_graph=second_order)     
        else:                                                                               
            encoder_copy.load_state_dict(fast_weights, strict=False)
            pred = encoder_copy(D1)                              
            loss1 = loss_fn(pred, D1_label)
            grads = torch.autograd.grad(loss1, encoder_copy.parameters(), create_graph=second_order)    # 不能基于fast_weights.values()求grad了
        fast_weights = OrderedDict((name, param - INNER_LR * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))  

    encoder_copy.load_state_dict(fast_weights, strict=False)
    pred = encoder_copy(D2)
    loss2 = loss_fn(pred, D2_label)
    _, pred_y = F.softmax(pred,dim=1).max(1)
    right_num = torch.eq(pred_y, D2_label).sum().cpu().numpy()
    
    return  loss2, right_num


def inner_update_2(encoder,loss_fn,device,second_order):
    # get task data: D1 for train, D2 for 'test'
    D1, D1_label, D2, D2_label = get_random_data(device)
    # get model parameters, then manually update
    fast_weights = OrderedDict((name, param) for (name, param) in encoder.named_parameters())  

    for i in range(INNER_UPDATE_NUM):
        if i == 0:
            pred = encoder(D1)
            loss1 = loss_fn(pred, D1_label)                 
            grads = torch.autograd.grad(loss1, encoder.parameters(), create_graph=second_order)     
        else:                                                                               
            encoder.weights_reset(fast_weights) 
            pred = encoder(D1)                               
            loss1 = loss_fn(pred, D1_label)
            grads = torch.autograd.grad(loss1, encoder.parameters(), create_graph=second_order)    # 不能基于fast_weights.values()求grad了
        fast_weights = OrderedDict((name, param - INNER_LR * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))  

    encoder.weights_reset(fast_weights) 
    pred = encoder(D2)
    loss2 = loss_fn(pred, D2_label)
    _, pred_y = F.softmax(pred,dim=1).max(1)
    right_num = torch.eq(pred_y, D2_label).sum().cpu().numpy()
    
    return  loss2, right_num


def inner_update_3(encoder,loss_fn,device,second_order):
    # get task data: D1 for train, D2 for 'test'
    D1, D1_label, D2, D2_label = get_random_data(device)
    # get model parameters, then manually update
    fast_weights = OrderedDict((name, param) for (name, param) in encoder.named_parameters()) 

    for i in range(INNER_UPDATE_NUM):
        if i == 0:
            pred = encoder(D1)
            loss1 = loss_fn(pred, D1_label)                 
            grads = torch.autograd.grad(loss1, encoder.parameters(), create_graph=second_order)     
        else:                                                                               
            handler_ls = encoder.forward_hooks(fast_weights)
            pred = encoder(D1) 
            remove_handler(handler_ls)              
            loss1 = loss_fn(pred, D1_label)
            grads = torch.autograd.grad(loss1, fast_weights.values(), retain_graph=second_order)
        fast_weights = OrderedDict((name, param - INNER_LR * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))  

    handler_ls = encoder.forward_hooks(fast_weights)
    pred = encoder(D2) 
    remove_handler(handler_ls) 
    loss2 = loss_fn(pred, D2_label)
    _, pred_y = F.softmax(pred,dim=1).max(1)
    right_num = torch.eq(pred_y, D2_label).sum().cpu().numpy()
    
    return  loss2, right_num


def main():

    # 是否开启二阶梯度的计算
    second_order = False

    # init neural network, optimizer, lr_scheduler, loss_function
    model = CNNEncoder(CLASS_NUM, num_channel, h_dim, embed_dim)
    optimizer_out = torch.optim.SGD(model.parameters(), lr=META_LR, momentum=0.9) 
    loss_fn = CrossEntropyLoss()

    # 支持使用CPU和单GPU
    if (gpu_id is not None) and torch.cuda.is_available():
        device = torch.device("cuda:%d"%gpu_id[0])
        if len(gpu_id)>1:       
            raise ValueError('only support single GPU')
        elif len(gpu_id)==1:    
            model = model.to(device)
    else:
        device=torch.device("cpu")

    model.train()
    # meta optimization
    for i in tqdm(range(META_TRAIN_ITER)):
        list_loss = []
        count_correct = 0
        
        for j in range(BATCH_TASK_NUM):
            loss2, right_num = inner_update_1(model,loss_fn,device,second_order) # _1 _2 _3
            list_loss.append(loss2)
            count_correct += right_num

        sumloss = sum(list_loss)/len(list_loss)   
        optimizer_out.zero_grad()     
        sumloss.backward()       
        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer_out.step()
        
        # show message
        if (i+1)%100 == 0:
            print_str = 'Iter: %d, loss = %.4f, train_acc = %.4f'%(i+1, sumloss.item(), count_correct/1.0/BATCH_TASK_NUM/BATCH_SIZE)
            print(print_str)


if __name__ == '__main__':
    main()

