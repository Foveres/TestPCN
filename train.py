import open3d as o3d
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
from dataset import *
from model import *
from utils import *
import os
import json
import time, datetime
import visdom
from time import time

sys.path.append("./emd/")
import emd_module as emd

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')  # batchSize：一次训练所读取的样本数
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)   # num_workers是加载数据（batch）的线程数目
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')  # 要训练的次数
parser.add_argument('--model', type=str, default='', help='optional reload model path')  # 可选重新加载模型路径
parser.add_argument('--num_points', type=int, default=8192, help='number of points')  # 点的个数
parser.add_argument('--n_primitives', type=int, default=16, help='number of surface elements')  # 平面元素个数
parser.add_argument('--env', type=str, default="MSN_TRAIN", help='visdom environment')  # 可视化环境

opt = parser.parse_args()
print(opt)


class FullModel(nn.Module):
    def __init__(self, model1, model2, model3):
        super(FullModel, self).__init__()
        self.model_1 = model1 # MSN 网络模型
        self.model_2 = model2 # MSN 网络模型
        self.model_3 = model3 # MSN 网络模型
        self.EMD = emd.emdModule()

    def forward(self, inputs, gt, eps, iters):  # torch.Size([4, 3, 5000]) torch.Size([4, 8192, 3])
        output1, expansion_penalty_1 = self.model_1(inputs)     # 5000——6144 torch.Size([4, 6120, 3])
        output2, expansion_penalty_2 = self.model_2(output1.transpose(1, 2))    # 6144——7168 torch.Size([4, 7168, 3])
        output3, expansion_penalty_3 = self.model_3(output2.transpose(1, 2))    # 7168——8192 torch.Size([4, 8192, 3])

        gt1_idx = farthest_point_sample(gt, 6144, RAN=False)  # torch.Size([4, 6144])
        gt1 = index_points(gt, gt1_idx)  # torch.Size([4, 6144, 3])
        gt1 = Variable(gt1, requires_grad=False)  # torch.Size([4, 6144, 3])

        gt2_idx = farthest_point_sample(gt, 7168, RAN=False)  # torch.Size([4, 7168])
        gt2 = index_points(gt, gt2_idx)  # torch.Size([4, 7168, 3])
        gt2 = Variable(gt2, requires_grad=False)  # torch.Size([24, 64, 3])

        gt3 = gt[:, :, :3]   # torch.Size([4, 8192, 3])

        dist, _ = self.EMD(output1, gt1, eps, iters)
        emd1 = torch.sqrt(dist).mean(1)

        dist, _ = self.EMD(output2, gt2, eps, iters)
        emd2 = torch.sqrt(dist).mean(1)

        dist, _ = self.EMD(output3, gt3, eps, iters)
        emd3 = torch.sqrt(dist).mean(1)

        return output1, emd1, expansion_penalty_1, output2, emd2, expansion_penalty_2, output3, emd3, expansion_penalty_3

# 设置visdom查看训练结果
vis = visdom.Visdom(port=8097, env=opt.env)  # set your port
# 设置开始时间
now = datetime.datetime.now()
# 设置模型保存路径
save_path = now.isoformat()
# 新建日志
if not os.path.exists('./log/'):
    os.mkdir('./log/')
# 日志文件：目录
dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
# 日志文件：文件名字
logname = os.path.join(dir_name, 'log.txt')
# 将文件复制到dir_name目录下
os.system('cp ./train.py %s' % dir_name)
os.system('cp ./dataset.py %s' % dir_name)
os.system('cp ./model.py %s' % dir_name)

# 设置随机种子
opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10

# （第一步：数据处理）设置数据集
dataset = ShapeNet(train=True, npoints=opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers),drop_last=True)

dataset_test = ShapeNet(train=False, npoints=opt.num_points)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers),drop_last=True)
# 查看数据集的大小
len_dataset = len(dataset)
print("Train Set Size: ", len_dataset)

# 第二步：网络结构。新建网络、分布式训练、放到GPU上、权重初始化
network1 = BasicNet(num_points=6144,n_primitives=12)
network2 = BasicNet(num_points=7168,n_primitives=14)
network3 = BasicNet(num_points=8192,n_primitives=16)
network = torch.nn.DataParallel(FullModel(network1,network2,network3))
network.cuda()
network.apply(weights_init)  # initialization of the weight

# 如果默认模型是空
if opt.model != '':
    # network.module.model.load_state_dict(torch.load(opt.model))
    network.load_state_dict(torch.load(opt.model))
    print("Previous weight loaded ")

# 第四步：优化器，参数有网络的参数和learning rate
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)

train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f:  # open and append
    # f.write(str(network.module.model) + '\n')
    f.write(str(network) + '\n')

train_curve = []
val_curve = []
labels_generated_points = torch.Tensor(
    range(1, (opt.n_primitives + 1) * (opt.num_points // opt.n_primitives) + 1)).view(
    opt.num_points // opt.n_primitives, (opt.n_primitives + 1)).transpose(0, 1)
labels_generated_points = (labels_generated_points) % (opt.n_primitives + 1)
labels_generated_points = labels_generated_points.contiguous().view(-1)

for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_loss.reset()
    # network.module.model.train()  # 使用在训练时：启用 BatchNormalization 和 Dropout
    network.train()

    # learning rate schedule
    if epoch == 20:
        # optimizer = optim.Adam(network.module.model.parameters(), lr=lrate / 10.0)
        optimizer = optim.Adam(network.parameters(), lr=lrate / 10.0)
    if epoch == 40:
        optimizer = optim.Adam(network.parameters(), lr=lrate / 100.0)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()   # 梯度清0
        id, input, gt = data    # input torch.Size([8, 5000, 3])  gt torch.Size([8, 8192, 3])
        input = input.float().cuda()
        gt = gt.float().cuda()
        input = input.transpose(2, 1).contiguous() # contiguous()相当于做了个深度拷贝修改一个值，另一个不改动。input torch.Size([8, 3, 5000])

        output1, emd1, expansion_penalty_1, output2, emd2, expansion_penalty_2, output3, emd3, expansion_penalty_3 = network(input, gt.contiguous(), 0.005, 50)    # 数据输入网络 output1 torch.Size([8, 8192, 3]) output2  torch.Size([8, 8192, 3])
        loss_net = emd1.mean() + emd2.mean() + emd3.mean() + expansion_penalty_1.mean() * 0.1 + expansion_penalty_2.mean() * 0.1 + expansion_penalty_3.mean() * 0.1       # 损失函数
        loss_net.backward()     # (损失的反向传播)
        optimizer.step()        # (更新参数)

        if i % 10 == 0:
            idx = random.randint(0, input.size()[0] - 1)
            vis.scatter(X=gt.contiguous()[idx].data.cpu()[:, :3],
                        win='TRAIN_GT',
                        opts=dict(
                            title=id[idx],
                            markersize=2,
                        ),
                        )
            vis.scatter(X=input.transpose(2, 1).contiguous()[idx].data.cpu(),
                        win='TRAIN_INPUT',
                        opts=dict(
                            title=id[idx],
                            markersize=2,
                        ),
                        )
            vis.scatter(X=output1[idx].data.cpu(),
                        Y=labels_generated_points[0:output1.size(1)],
                        win='TRAIN_COARSE',
                        opts=dict(
                            title=id[idx],
                            markersize=2,
                        ),
                        )
            vis.scatter(X=output2[idx].data.cpu(),
                        Y=labels_generated_points[0:output2.size(1)],
                        win='TRAIN_COARSE',
                        opts=dict(
                            title=id[idx],
                            markersize=2,
                        ),
                        )
            vis.scatter(X=output3[idx].data.cpu(),
                        Y=labels_generated_points[0:output3.size(1)],
                        win='TRAIN_OUTPUT',
                        opts=dict(
                            title=id[idx],
                            markersize=2,
                        ),
                        )
        print(opt.env + ' train [%d: %d/%d]  emd1: %f emd2: %f  emd3: %f expansion_penalty_1: %f expansion_penalty_2: %f expansion_penalty_3: %f' % (
            epoch, i, len_dataset / opt.batchSize, emd1.mean().item(), emd2.mean().item(), emd3.mean().item(),
            expansion_penalty_1.mean().item(),expansion_penalty_2.mean().item(),expansion_penalty_3.mean().item()))
        train_curve.append(train_loss.avg)

    # VALIDATION
    if epoch % 5 == 0:
        val_loss.reset()
        network.module.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_test, 0):
                id, input, gt = data
                input = input.float().cuda()
                gt = gt.float().cuda()
                input = input.transpose(2, 1).contiguous()
                output1, emd1, expansion_penalty = network(input, gt.contiguous(), 0.004, 3000)
                val_loss.update(emd1.mean().item())
                idx = random.randint(0, input.size()[0] - 1)
                vis.scatter(X=gt.contiguous()[idx].data.cpu()[:, :3],
                            win='VAL_GT',
                            opts=dict(
                                title=id[idx],
                                markersize=2,
                            ),
                            )
                vis.scatter(X=input.transpose(2, 1).contiguous()[idx].data.cpu(),
                            win='VAL_INPUT',
                            opts=dict(
                                title=id[idx],
                                markersize=2,
                            ),
                            )
                vis.scatter(X=output1[idx].data.cpu(),
                            Y=labels_generated_points[0:output1.size(1)],
                            win='VAL_COARSE',
                            opts=dict(
                                title=id[idx],
                                markersize=2,
                            ),
                            )
                vis.scatter(X=output2[idx].data.cpu(),
                            Y=labels_generated_points[0:output2.size(1)],
                            win='VAL_COARSE',
                            opts=dict(
                                title=id[idx],
                                markersize=2,
                            ),
                            )
                vis.scatter(X=output3[idx].data.cpu(),
                            Y=labels_generated_points[0:output3.size(1)],
                            win='VAL_OUTPUT',
                            opts=dict(
                                title=id[idx],
                                markersize=2,
                            ),
                            )
                print(opt.env + ' val [%d: %d/%d]  emd1: %f emd2: %f emd3: %f expansion_penalty_1: %f expansion_penalty_2: %f expansion_penalty_3: %f' % (
                    epoch, i, len_dataset / opt.batchSize, emd1.mean().item(), emd2.mean().item(), emd2.mean().item(),
                    expansion_penalty_1.mean().item(), expansion_penalty_2.mean().item(), expansion_penalty_3.mean().item()))

    val_curve.append(val_loss.avg)

    vis.line(X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
             Y=np.column_stack((np.array(train_curve), np.array(val_curve))),
             win='loss',
             opts=dict(title="emd", legend=["train_curve" + opt.env, "val_curve" + opt.env], markersize=2, ), )
    vis.line(X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
             Y=np.log(np.column_stack((np.array(train_curve), np.array(val_curve)))),
             win='log',
             opts=dict(title="log_emd", legend=["train_curve" + opt.env, "val_curve" + opt.env], markersize=2, ), )

    log_table = {
        "train_loss": train_loss.avg,
        "val_loss": val_loss.avg,
        "epoch": epoch,
        "lr": lrate,
        "bestval": best_val_loss,

    }
    with open(logname, 'a') as f:
        f.write('json_stats: ' + json.dumps(log_table) + '\n')

    print('saving net...')
    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
