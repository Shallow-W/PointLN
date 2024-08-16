import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

warnings.filterwarnings('ignore')  # 忽略警告消息
CLASS_NUM = 20  # （使用自己的数据集时需要更改）

class plnLoss(nn.Module):
    def __init__(self, S, B, w_class, w_coord, w_link):
        super(plnLoss, self).__init__()
        self.S = S  # S = 14
        self.B = B  # B = 2
        # 权重系数
        self.w_class = w_class
        # 权重系数
        self.w_coord = w_coord
        self.w_link = w_link
        self.classes = 20
        self.w_pt = 1
        self.w_nopt = 0.05
        self.grid_size = 14

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,14,14,204)
        target_tensor: (tensor) size(batchsize,14,14,204)，就是在yoloData中制作的标签
        按照pij,xij,yij,lxij,lyij,qij的顺序 51维
        pij 0 xij 1 yij 2
        lxij 3-16 lyij 17-30 qij 31-50
        0-50=51
        设置pij为1，即存在点
        '''
        pred_tensor = pred_tensor.permute(0, 2, 3, 1)
        # batchsize大小
        # print(pred_tensor.size(),target_tensor.size())
        N = pred_tensor.size()[0]
        # 分成中心点和角点
        target_tensor_center = target_tensor[:, :, :, 0:102]
        target_tensor_corner = target_tensor[:,:,:,102:]
        pred_tensor_center = pred_tensor[:,:,:,0:102]
        pred_tensor_corner = pred_tensor[:,:,:,102:]
        # 判断目标是否在网络内
        coo_mask_center = target_tensor_center[:, :, :, 0] > 0
        coo_mask_corner = target_tensor_corner[:, :, :, 0] > 0
        # 判断目标不在那个网格，输出B*14*14的矩阵，没有目标的地方为True，其他地方为false
        noo_mask_center = target_tensor_center[:, :, :, 0] == 0
        noo_mask_corner = target_tensor_corner[:, :, :, 0] == 0
        # print("coo_mask_center",coo_mask_center.shape)
        # print("noo_mask",noo_mask.shape)
        # 将 coo_mask_center tensor 在最后一个维度上增加一维，并将其扩展为与 target_tensor tensor 相同的形状，得到含物体的坐标等信息，大小为batchsize*14*14*204
        coo_mask_center = coo_mask_center.unsqueeze(-1).expand_as(target_tensor_center)
        coo_mask_corner = coo_mask_corner.unsqueeze(-1).expand_as(target_tensor_corner)
        # 将 noo_mask 在最后一个维度上增加一维，并将其扩展为与 target_tensor tensor 相同的形状，得到不含物体的坐标等信息，大小为batchsize*14*14*204
        noo_mask_center = noo_mask_center.unsqueeze(-1).expand_as(target_tensor_center)
        noo_mask_corner = noo_mask_corner.unsqueeze(-1).expand_as(target_tensor_corner)
        # print('coo_mask_center',coo_mask_center.shape)
        # print('shape:',pred_tensor[coo_mask_center].shape)
        # # 类别信息
        # class_pred = coo_pred[:, 10:]  # [n_coord, 20]
        # 根据label的信息从预测的张量取出对应位置的网格的30个信息按照出现序号拼接成以一维张量
        # 所有的box的位置坐标和置信度放到pred中，塑造成X行51列（-1表示自动计算），一个box包含51个值
        # 拆分成多个点的预测51列
        coo_pred_center = pred_tensor_center[coo_mask_center].view(-1, int(102))
        point_pred_center = coo_pred_center[:, :].contiguous().view(-1, 51)
        coo_pred_corner = pred_tensor_corner[coo_mask_corner].view(-1, int(102))
        point_pred_corner = coo_pred_corner[:, :].contiguous().view(-1, 51)
        # 所有的box的位置坐标和置信度放到target中，塑造成X行51列（-1表示自动计算），一个box包含51个值
        coo_target_center = target_tensor_center[coo_mask_center].view(-1, int(102))
        point_target_center = coo_target_center[:, :].contiguous().view(-1, 51)
        coo_target_corner = target_tensor_corner[coo_mask_corner].view(-1, int(102))
        point_target_corner = coo_target_corner[:, :].contiguous().view(-1, 51)
        # 不包含物体grid ceil的置信度损失,这里是label的输出的向量。
        noo_pred_center = pred_tensor_center[noo_mask_center].view(-1, int(102))
        noo_target_center = target_tensor_center[noo_mask_center].view(-1, int(102))
        noo_pred_corner = pred_tensor_corner[noo_mask_corner].view(-1, int(102))
        noo_target_corner = target_tensor_corner[noo_mask_corner].view(-1, int(102))
        # 创建一个跟noo_pred相同形状的张量，形状为（x,204），里面都是全0或全1，再使用bool将里面的0或1转为true和false
        noo_pred_mask_center = torch.cuda.ByteTensor(noo_pred_center.size()).bool()
        noo_pred_mask_corner = torch.cuda.ByteTensor(noo_pred_corner.size()).bool()
        # 把创建的noo_pred_mask全部改成false，因为这里对应的是没有目标的张量
        noo_pred_mask_center.zero_()
        noo_pred_mask_corner.zero_()
        # 把不包含目标的张量的置信度位置置为1
        noo_pred_mask_center[:, 0] = 1
        noo_pred_mask_center[:, 0 + 51 * 1] = 1
        noo_pred_mask_corner[:, 0 ] = 1
        noo_pred_mask_corner[:, 0 + 51 * 1] = 1
        # 设置成GPU
        noo_pred_center = noo_pred_center.to(noo_pred_mask_center.device)
        noo_target_center = noo_target_center.to(noo_pred_mask_center.device)
        noo_pred_corner = noo_pred_corner.to(noo_pred_mask_corner.device)
        noo_target_corner = noo_target_corner.to(noo_pred_mask_corner.device)
        # 把不包含目标的置信度提取出来拼接成一维张量
        noo_pred_c_center = noo_pred_center[noo_pred_mask_center]
        noo_pred_c_corner = noo_pred_corner[noo_pred_mask_corner]
        # 同noo_pred_c
        noo_target_c_center = noo_target_center[noo_pred_mask_center]
        noo_target_c_corner = noo_target_corner[noo_pred_mask_corner]

        # 计算loss，让预测的值越小越好，因为不包含目标，置信度越为0越好
        # 计算中心点和角点的 noo0bj误差
        nooobj_loss = F.mse_loss(noo_pred_c_center, noo_target_c_center, size_average=False)  # 均方误差
        nooobj_loss += F.mse_loss(noo_pred_c_corner, noo_target_c_corner, size_average=False)  # 均方误差
        # print("noobj_loss", nooobj_loss)
        """
        计算包含目标的损失：位置损失+类别损失
        在pln中，需要有(pij-1)**2 概率损失，wclass类别损失，wcoord位置损失，wlink，链接损失
        """
        # 创建一跟box_target相同的张量,这里用来匹配后面负责预测的框
        coo_response_mask_center = torch.cuda.ByteTensor(point_target_center.size()).bool()
        coo_response_mask_corner = torch.cuda.ByteTensor(point_target_corner.size()).bool()
        # 全部置为False
        coo_response_mask_center.zero_()  # 全部元素置False
        coo_response_mask_corner.zero_()
        # 创建多个张量，用来匹配xij,yij,lxij,lyij,qij列
        coo_response_mask_pij = coo_response_mask_center.clone()
        coo_response_mask_xy = coo_response_mask_center.clone()
        coo_response_mask_lxy = coo_response_mask_center.clone()
        coo_response_mask_qij = coo_response_mask_center.clone()
        coo_response_mask_pij_corner = coo_response_mask_corner.clone()
        coo_response_mask_xy_corner = coo_response_mask_corner.clone()
        coo_response_mask_lxy_corner = coo_response_mask_corner.clone()
        coo_response_mask_qij_corner = coo_response_mask_corner.clone()

        # 设备GPU
        point_pred_center = point_pred_center.to(coo_response_mask_center.device)
        point_target_center = point_target_center.to(coo_response_mask_center.device)
        point_pred_corner = point_pred_corner.to(coo_response_mask_corner.device)
        point_target_corner = point_target_corner.to(coo_response_mask_corner.device)

        # 计算pij误差
        coo_response_mask_pij[:,0] = 1
        coo_response_mask_pij_corner[:,0] = 1
        pred_c = point_pred_center[coo_response_mask_pij]
        target_c = point_target_center[coo_response_mask_pij]
        p_loss = F.mse_loss(pred_c,target_c,size_average=False)
        pred_c = point_pred_corner[coo_response_mask_pij_corner]
        target_c = point_target_corner[coo_response_mask_pij_corner]
        p_loss += F.mse_loss(pred_c, target_c, size_average=False)
        # 计算xij，yij误差
        coo_response_mask_xy[:,1:3] = 1
        coo_response_mask_xy_corner[:,1:3] = 1
        pred_c = point_pred_center[coo_response_mask_xy]
        target_c = point_target_center[coo_response_mask_xy]
        wcoord_loss = F.mse_loss(pred_c,target_c,size_average=False)
        pred_c = point_pred_corner[coo_response_mask_xy_corner]
        target_c = point_target_corner[coo_response_mask_xy_corner]
        wcoord_loss += F.mse_loss(pred_c, target_c, size_average=False)
        # 计算l误差
        coo_response_mask_lxy[:,3:31] = 1
        coo_response_mask_lxy_corner[:,3:31] = 1
        pred_c = point_pred_center[coo_response_mask_lxy]
        target_c = point_target_center[coo_response_mask_lxy]
        wlink_loss = F.mse_loss(pred_c, target_c, size_average=False)
        pred_c = point_pred_corner[coo_response_mask_lxy_corner]
        target_c = point_target_corner[coo_response_mask_lxy_corner]
        wlink_loss += F.mse_loss(pred_c, target_c, size_average=False)
        # 计算qij误差
        coo_response_mask_qij[:,31:] = 1
        coo_response_mask_qij_corner[:,31:] = 1
        pred_c = point_pred_center[coo_response_mask_qij]
        target_c = point_target_center[coo_response_mask_qij]
        wclass_loss = F.mse_loss(pred_c, target_c, size_average=False)
        pred_c = point_pred_corner[coo_response_mask_qij_corner]
        target_c = point_target_corner[coo_response_mask_qij_corner]
        wclass_loss += F.mse_loss(pred_c, target_c, size_average=False)

        return p_loss + self.w_coord * wcoord_loss + self.w_class * wclass_loss + self.w_link * wlink_loss + nooobj_loss * 0.04
