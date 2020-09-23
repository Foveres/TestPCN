from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys

sys.path.append("./expansion_penalty/")
import expansion_penalty_module as expansion

sys.path.append("./MDS/")
import MDS_module


class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=8192, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  # 输入通道是3，输出通道是64，卷积核大小是1，默认0填充，步长为1
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size  # 1026
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)  # 1026 1026 1
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)  # 1026 513 1
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)  # 513 256 1
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)  # 256 3 1

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)  # 1026
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)  # 513
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)  # 256

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # 4 1026 512
        x = F.relu(self.bn2(self.conv2(x)))  # 4 513 512
        x = F.relu(self.bn3(self.conv3(x)))  # torch.Size([4, 256, 512])
        x = self.th(self.conv4(x))  # torch.Size([4, 3, 512])
        return x


class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x


class BasicNet(nn.Module):
    def __init__(self, num_points=8192, bottleneck_size=1024, n_primitives=16):
        super(BasicNet, self).__init__()
        self.num_points = num_points  # 8192
        self.bottleneck_size = bottleneck_size  # 1024
        self.n_primitives = n_primitives  # 16
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True),  # 8192
            nn.Linear(1024, self.bottleneck_size),  # 1024 1024
            nn.BatchNorm1d(self.bottleneck_size),  # 1024
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()

    def forward(self, x):
        partial = x  # x：torch.Size([4, 3, 5000])
        x = self.encoder(x)  # 输出的x：[4,1024]
        outs = []
        for i in range(0, self.n_primitives):  # 【K的值0-15】self.n_primitives
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2,
                                                        self.num_points // self.n_primitives))  # rand_grid创建了一个[4,2,512],4个2*512矩阵，值为0
            rand_grid.data.uniform_(0, 1)  # 这个就是个初始化，uniform_指的是均匀分布。使这个值在0-1间均匀分布
            # expand()这个函数的作用就是对指定的维度进行数值大小的改变。只能改变维大小为1的维，否则就会报错。不改变的维可以传入 - 1
            # 或者原来的数值。
            y = x.unsqueeze(2).expand(x.size(0), x.size(1),
                                      rand_grid.size(2)).contiguous()  # x在第二个轴扩展一个轴[4,1024,512],跳出循环继续下一次循环
            # 调用contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据，并会真正改变Tensor的内容，按照变换之后的顺序存放数据。
            y = torch.cat((rand_grid, y), 1).contiguous()  # rand_grid和y按维数1拼接
            outs.append(self.decoder[i](y))  # torch.Size([4, 3, 512])  self.decoder[i](y)

        outs = torch.cat(outs, 2).contiguous()
        out1 = outs.transpose(1, 2).contiguous()  # 粗的输出

        dist, _, mean_mst_dis = self.expansion(out1, self.num_points // self.n_primitives, 1.5)  # 粗输出的扩展惩罚
        loss_mst = torch.mean(dist)

        return out1, loss_mst  # out1：torch.Size([4, 8192, 3])；  tout2：orch.Size([4, 8192, 3])