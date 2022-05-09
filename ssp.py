import os
import threading
from datetime import datetime
from multiprocess import Process,Pool,Manager
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim
import torchvision
from torch.utils.data import DataLoader,SubsetRandomSampler
import time
batch_size = 8
image_w = 64
image_h = 64
num_classes = 10
batch_update_size = 1
num_batches = 10
loss_fn = nn.CrossEntropyLoss()
one_hot_indices = torch.LongTensor(batch_size) .random_(0, num_classes).view(batch_size, 1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize(64)])
 # 加载CIFAR10数据,CIFAR100数据
test_set = datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
# test_set = datasets.CIFAR100(root='./', train=False, download=True, transform=transform)

all_length = 10000
indices = list(range(len(test_set)))
split_valid = int(np.floor(1000))
train_idx = indices[:split_valid]
train_sampler = SubsetRandomSampler(train_idx)
train_loader = DataLoader(test_set, num_workers=1, batch_size=batch_size, sampler=train_sampler)


def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")

# 批量更新参数server
class BatchUpdateParameterServer(object):

    # 初始化并构建模型
    def __init__(self, batch_update_size=batch_update_size):
        self.model = torchvision.models.resnet50(num_classes=num_classes)
        self.lock = threading.Lock() # 加锁，以便于后续同步更新参数
        self.future_model = torch.futures.Future() #用来处理更新后的模型
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    # 返回init函数中构建好的模型
    def get_model(self):
        return self.model


    """
    批处理RPC
    同步批量更新参数服务器:使用@rpc.functions。用于参数更新和检索的Async_execution。
    """

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, name):
        stale = 1
        self = ps_rref.local_value()
        timed_log(f"PS got {self.curr_update_size}/{batch_update_size} updates {name}")
        for p, g in zip(self.model.parameters(), grads):
            p.grad += g #累积梯度
        with self.lock: # 判断有锁的时候开始批量更新参数，保证同步进行
            self.curr_update_size += 1
            fut = self.future_model
            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size
                self.curr_update_size = 0
                self.optimizer.step() #最后通过梯度下降执行一步参数更新
                self.optimizer.zero_grad() #把梯度置零，也就是把loss关于weight的导数变成0
                fut.set_result(self.model) # 将更新后的模型发送给所有训练者
                timed_log("PS updated model")
                self.future_model = torch.futures.Future()# 使用更新后的模型来设置future_model

        return fut # 该对象将被用来处理更新后的模型


class Trainer(object):

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.MSELoss()
        self.one_hot_indices = torch.LongTensor(batch_size) \
                                    .random_(0, num_classes) \
                                    .view(batch_size, 1)

# 获取下一批，遍历批数，获取input和lable
def get_next_batch():
    for i in range(num_batches):
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) .scatter_(1, one_hot_indices, 1)
        yield inputs.cpu(), labels.cpu(),i

def train1(m,optimizer,name,mylist,loss1,loss2,loss3,loss4,loss5,loss6):
    # name = rpc.get_worker_info().name
    future_model = torch.futures.Future()
    for i in range(30):
        b = 0
        lo1 = 0
        for inputs, labels in train_loader:     # 获取下一批，遍历批数，获取input和lable
            timed_log(f"{name} processing one batch")
            loss = loss_fn(m(inputs), labels)
            loss.backward() #反向传播计算得到每个参数的梯度值
            timed_log(f"{name} reporting grads")
            fut = future_model
            optimizer.step()
            optimizer.zero_grad()
            fut.set_result(m)
            future_model = torch.futures.Future()
            timed_log(f"{name} got updated model")
            b = b + 1
            lo1 = loss.item() + lo1
            loss_all = lo1 / b
            a1, a2, a3, a4, a5, a6 = 0, 0, 0, 0, 0, 0
            for j in mylist:
                if (j == 1):
                    a1 = a1 + 1
                if (j == 2):
                    a2 = a2 + 1
                if (j == 3):
                    a3 = a3 + 1
                if (j == 4):
                    a4 = a4 + 1
                if (j == 5):
                    a5 = a5 + 1
                if (j == 6):
                    a6 = a6 + 1
            if (np.abs(a1 - a2) > 5 or np.abs(a1 - a3) > 5 or np.abs(a1 - a4) > 5 or np.abs(a1 - a5) > 5 or np.abs(
                    a1 - a6) > 5 or np.abs(a3 - a2) > 5 or np.abs(a4 - a2) > 5 or np.abs(a5 - a2) > 5
                    or np.abs(a6 - a2) > 5 or np.abs(a3 - a4) > 5 or np.abs(a3 - a5) > 5 or np.abs(
                        a3 - a6) > 5 or np.abs(a4 - a5) > 5 or np.abs(a4 - a6) > 5):
                time.sleep(0.001)
        if(name==1):
            mylist.append(1)
            loss1.append(loss_all)
        if (name == 2):
            mylist.append(2)
            loss2.append(loss_all)
        if (name == 3):
            mylist.append(3)
            loss3.append(loss_all)
        if (name == 4):
            mylist.append(4)
            loss4.append(loss_all)
        if (name == 5):
            mylist.append(5)
            loss5.append(loss_all)
        if (name == 6):
            mylist.append(6)
            loss6.append(loss_all)

    return fut

def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)
    # trainer.train() # 调用 Trainer 的方法
    manager = Manager()
    mylist = manager.list()
    loss1 = manager.list()
    loss2 = manager.list()
    loss3 = manager.list()
    loss4 = manager.list()
    loss5 = manager.list()
    loss6 = manager.list()

    pool = Pool(processes=6)
    # model1 = torchvision.models.vgg16(num_classes=num_classes)
    # model1 = torchvision.models.vgg11(num_classes=num_classes)
    # model1 = torchvision.models.resnet50(num_classes=num_classes)
    model1 = torchvision.models.resnet18(num_classes=num_classes)
    model2 = torchvision.models.resnet18(num_classes=num_classes)
    model3 = torchvision.models.resnet18(num_classes=num_classes)
    model4 = torchvision.models.resnet18(num_classes=num_classes)
    model5 = torchvision.models.resnet18(num_classes=num_classes)
    model6 = torchvision.models.resnet18(num_classes=num_classes)
    modeli = [model1,model2,model3,model4,model5,model6]

    for i in range(1,7):
        model = modeli[i-1]
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        li = pool.apply_async(train1,(model,optimizer,i,mylist,loss1,loss2,loss3,loss4,loss5,loss6))
    pool.close()
    pool.join()
    print('list',mylist)
    print('loss1',loss1)
    print('loss2',loss2)
    print('loss3',loss3)
    print('loss4',loss4)
    print('loss5',loss5)
    print('loss6',loss6)

    timed_log("Finish training")


"""
rpc_async -> 训练
"""
#远程过程调用协议
def run_ps(trainers):
    timed_log("Start training")
    ps_rref = rpc.RRef(BatchUpdateParameterServer())
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,)) )# 运行run_trainer)



def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0  # infinite timeout
     )
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            #rpc_backend_options=options
        )
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc( # 参数服务器
            "ps",
            rank=rank, #rank：分配给进程组中每个进程的唯一标识符，这个 worker 是全局第几个 worker；
            world_size=world_size, #进程组中的进程数，可以认为是全局进程个数，即总共有几个 worker;
            #rpc_backend_options=options
        )
        run_ps([f"trainer{r}" for r in range(1, world_size)])

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = 2
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)  #nprocs: 派生进程个数；join: 是否加入同一进程池；
