import os
import os.path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import plnLoss
from learn.PLN2.new_resnet import inceptionresnetv2

pd.set_option('display.max_rows', None)  # ÏÔÊ¾È«²¿ÐÐ
pd.set_option('display.max_columns', None)  # ÏÔÊ¾È«²¿ÁÐ

# 根据txt文件制作ground truth
CLASS_NUM = 20  # 使用其他训练集需要更改


class plnDataset(Dataset):
    image_size = 448

    def __init__(self, img_root, list_file, train, transform):
        # 初始化参数
        self.root = img_root
        self.train = train
        self.transform = transform
        # 后续要提取txt文件的信息，分类后装入以下三个列表
        # 文件名
        self.fnames = []
        # 位置信息
        self.boxes = []
        # 类别信息
        self.labels = []
        # 网格大小
        self.S = 14
        # 候选框个数
        self.B = 2
        # 类别数目
        self.C = CLASS_NUM
        # 求均值用的
        self.mean = (123, 117, 104)
        # 打开文件，就是voctrain.txt或者voctest.txt文件
        file_txt = open(list_file)
        # 读取txt文件每一行
        lines = file_txt.readlines()
        # 逐行开始操作
        for line in lines:
            # 去除字符串开头和结尾的空白字符，然后按照空白字符（包括空格、制表符、换行符等）分割字符串并返回一个列表
            splited = line.strip().split()
            # 存储图片的名字
            self.fnames.append(splited[0])
            # 计算一幅图片里面有多少个bbox，注意voctrain.txt或者voctest.txt一行数据只有一张图的信息
            num_boxes = (len(splited) - 1) // 5
            # 保存位置信息
            box = []
            # 保存标签信息
            label = []
            # 提取坐标信息和类别信息
            for i in range(num_boxes):
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                # 提取类别信息，即是20种物体里面的哪一种  值域 0-19
                c = splited[5 + 5 * i]
                # 存储位置信息
                box.append([x, y, x2, y2])
                # 存储标签信息
                label.append(int(c))
            # 解析完所有行的信息后把所有的位置信息放到boxes列表中，boxes里面的是每一张图的坐标信息，也是一个个列表，即形式是[[[x1,y1,x2,y2],[x3,y3,x4,y4]],[[x5,y5,x5,y6]]...]这样的
            self.boxes.append(torch.Tensor(box))
            # 形式是[[0],[1,2],[3,4,5]...]，注意这里是标签，对应位整型数据，表示里面的物体标签
            self.labels.append(torch.LongTensor(label))
        # 统计图片数量
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        # 获取一张图像
        fname = self.fnames[idx]
        # 读取这张图像
        img = cv2.imread(os.path.join(self.root + fname))
        # 拷贝一份，避免在对数据进行处理时对原始数据进行修改
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        # print(boxes)
        # print(labels)
        # print("boxes:", boxes)
        # print("labels:", labels)

        """
        数据增强里面的各种变换用pytorch自带的transform是做不到的，因为对图片进行旋转、随即裁剪等会造成bbox的坐标也会发生变化，
        所以需要自己来定义数据增强,这里推荐使用功albumentations更简单便捷
        """
        # if self.train:
        #     img, boxes = self.random_flip(img, boxes)
        #     img, boxes = self.randomScale(img, boxes)
        #     img = self.randomBlur(img)
        #     img = self.RandomBrightness(img)
        #     # img = self.RandomHue(img)
        #     # img = self.RandomSaturation(img)
        #     img, boxes, labels = self.randomShift(img, boxes, labels)
        #     # img, boxes, labels = self.randomCrop(img, boxes, labels)
        #     获取图像高宽信息
        h, w, _ = img.shape
        # 归一化位置信息，.expand_as(boxes)的作用是将torch.Tensor([w, h, w, h])扩展成和boxes一样的维度，这样才能进行后面的归一化操作
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)  # 坐标归一化处理[0,1]，为了方便训练
        # print("one box:", boxes)

        # cv2读取的图像是BGR，转成RGB
        img = self.BGR2RGB(img)
        # 减去均值，帮助网络更快地收敛并提高其性能
        img = self.subMean(img, self.mean)
        # 调整图像到统一大小
        img = cv2.resize(img, (self.image_size, self.image_size))
        # 将图片标签编码到7x7*30的向量，也就是我们yolov1的最终的输出形式，这个地方不了解的可以去看看yolov1原理
        # target = self.encoder(boxes, labels)
        loader = Label_loader("", "", "", 0, 448, 14)
        target = []
        for i in range(4):
            t = loader.load_label(i,boxes,labels)
            target.append(t)
        # 得到一个[4,14,14,204]维的tensor
        target = torch.stack(target)

        # 进行数据增强操作
        for t in self.transform:
            img = t(img)
        # 返回一张图像和所有标注信息
        # torch.set_printoptions(profile="full")
        # print("target",target)

        # print("img",img.shape)
        # print("target",target.shape)
        # print("labels",labels.shape)
        return img, target

    def __len__(self):
        return self.num_samples
        # 编码图像标签为7x7*30的向量，输入的boxes为一张图的归一化形式位置坐标(X1,Y1,X2,Y2)

    def encoder(self, boxes, labels):
        # 对一张图片进行encoder处理
        # 网格大小
        grid_num = 14
        # 定义一个空的14*14*204的张量
        # 在pln中，应该是14*14*（（1+1+1+14+14+20）*4）=14x14x204，pij,xij,yij,Lxij,Lyij,qij(20种)
        # target = torch.zeros((grid_num, grid_num, int(CLASS_NUM + 10)))
        target = torch.zeros((grid_num, grid_num, int(CLASS_NUM + self.S * 2 + 3) * self.B * 2))
        # 对网格进行归一化操作
        cell_size = 1. / grid_num  # 1/14
        # 每一张图的标注框的中心点坐标，cxcy也是一个列表，形式为[[cx1,cy1],[cx2,cy2]...]
        # cxcy是一个列表，包含中心点的归一化坐标
        # right_top是一个列表，包含右上角点的归一化坐标
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        right_top = boxes[:, [2, 1]]
        # print(boxes)
        # tensor([[0.4180, 0.5428, 0.7760, 1.0000],
        #         [0.4040, 0.0053, 0.9980, 0.7513]])
        # print(cxcy)
        # tensor([[0.5970, 0.7714],
        #         [0.7010, 0.3783]])

        # 遍历每个每张图上的标注框信息，cxcy.size()[0]为标注框个数，即计算一张图上几个标注框
        for i in range(cxcy.size()[0]):
            # 取中心点坐标，cxcy_sample表示相对整张图左上角归一化坐标  cxcy_sample: tensor([0.4030, 0.7714])
            cxcy_sample = cxcy[i]
            right_top_sample = right_top[i]
            # print(cxcy_sample)
            # print(right_top_sample)
            # 中心点坐标获取后，计算这个标注框属于哪个grid cell，因为上面归一化了，这里要反归一化，还原回去，坐标从0开始，所以减1，注意坐标从左上角开始
            # ij表示grid cell 编号  ij:tensor[5,10]
            ij = (cxcy_sample / cell_size).ceil() - 1
            corner_ij = (right_top_sample / cell_size).ceil() - 1

            # print("corner_ij",corner_ij)
            # print("ij:",ij)
            # ij: tensor([8., 10.])
            # ij: tensor([9., 5.])
            # 把标注框框所在的gird cell的的两个bounding box置信度全部置为1，多少行多少列，多少通道的值置为1
            # target[int(ij[1]), int(ij[0]), 4] = 1
            # target[int(ij[1]), int(ij[0]), 4 + 5 * 1] = 1
            # 把标注框框所在的gird cell的的两个bounding box类别置为1，这样就完成了该标注框的label信息制作了
            # target[int(ij[1]), int(ij[0]), int(labels[i]) + 10] = 1

            # 预测框的中心点在图像中的绝对坐标（xy），归一化的,相对gird cell左上角的xy坐标，中心点和角点
            xy = ij * cell_size
            corner_xy = corner_ij * cell_size
            # print("cxcy_sample:", cxcy_sample)
            # print("xy:", xy)
            # cxcy_sample: tensor([0.4030, 0.7714])
            # xy: tensor([0.3571, 0.7143])

            # 按照pij,xij,yij,lxij,lyij,qij的顺序 51维
            # pij 0 xij 1 yij 2
            # lxij 3-16 lyij 17-30 qij 31-50
            # 0-50=51
            # 设置pij为1，即存在点
            target[int(ij[1]), int(ij[0]), 0] = 1
            target[int(ij[1]), int(ij[0]), 0 + 51 * 1] = 1
            target[int(corner_ij[1]), int(corner_ij[0]), 0 + 51 * 2] = 1
            target[int(corner_ij[1]), int(corner_ij[0]), 0 + 51 * 3] = 1
            # 将xij，yij，lxij，lyij存储到target张量中
            target[int(ij[1]), int(ij[0]), 1:3] = xy
            target[int(ij[1]), int(ij[0]), 1 + 51 * 1:3 + 51 * 1] = xy
            target[int(corner_ij[1]), int(corner_ij[0]), 1 + 51 * 2:3 + 51 * 2] = corner_xy
            target[int(corner_ij[1]), int(corner_ij[0]), 1 + 51 * 3:3 + 51 * 3] = corner_xy
            # lxij
            target[int(ij[1]), int(ij[0]), 3 + int(corner_ij[0])] = 1
            target[int(ij[1]), int(ij[0]), 3 + int(corner_ij[0]) + 51 * 1] = 1
            target[int(corner_ij[1]), int(corner_ij[0]), 3 + int(ij[0]) + 51 * 2] = 1
            target[int(corner_ij[1]), int(corner_ij[0]), 3 + int(ij[0]) + 51 * 3] = 1
            # lyij
            target[int(ij[1]), int(ij[0]), 17 + int(corner_ij[1])] = 1
            target[int(ij[1]), int(ij[0]), 17 + int(corner_ij[1]) + 51 * 1] = 1
            target[int(corner_ij[1]), int(corner_ij[0]), 17 + int(ij[1]) + 51 * 2] = 1
            target[int(corner_ij[1]), int(corner_ij[0]), 17 + int(ij[1]) + 51 * 3] = 1
            # qij 将每个点对应的label项置为1
            target[int(ij[1]), int(ij[0]), 31 + labels[i]] = 1
            target[int(ij[1]), int(ij[0]), 31 + labels[i] + 51 * 1] = 1
            target[int(corner_ij[1]), int(corner_ij[0]), 31 + labels[i] + 51 * 2] = 1
            target[int(corner_ij[1]), int(corner_ij[0]), 31 + labels[i] + 51 * 3] = 1

            # print(target[int(ij[1]),int(ij[0])])
            # print(target[int(corner_ij[1]),int(corner_ij[0])])
            # test:
            # print(int(ij[1]),int(ij[0]))
            # print(target[int(ij[1]), int(ij[0])])
            #
            # print(int(corner_ij[1]),int(corner_ij[0]))
            # print(target[int(corner_ij[1]), int(corner_ij[0])])
            #
            # torch.set_printoptions(profile="full")
            # print("ij",ij)
            # print("corner_ij",corner_ij)
            # print("center:", target[int(ij[1]), int(ij[0]), :])
            # print("corner:", target[int(corner_ij[1]), int(corner_ij[0]), :])

        # 返回制作的标签，可以看到，除了有标注框所在得到位置，其他地方全为0
        return target  # (xc,yc) = 7*7   (w,h) = 1*1

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr


class Label_loader():
    def __init__(self, train_dir_obj, test_dir_obj, loader_type, seed, pic_width, S=14) -> None:
        # open object json
        # if loader_type == 'train':
        #     with open(train_dir_obj, 'r') as file:
        #         data = json.load(file)
        # elif loader_type == 'eval':
        #     with open(test_dir_obj, 'r') as file:
        #         data = json.load(file)
        # get data below
        self.boxes = []
        self.labels = []
        self.difficulties = []
        self.loader_type = loader_type

        self.eval_label = []
        self.eval_box = []

        self.s = S
        self.classes = 20
        self.B = 2  # the number of objects
        self.infinite = 100000000  # infinite
        self.pic_width = pic_width

        # random.seed(seed)  # set random seed
        torch.manual_seed(seed)

    def pij_pro(self, branch,boxes):
        posi = []
        posi_ct = []

        # 角点+中心点
        p_tensor = torch.zeros((self.s, self.s, 1))
        p_ct_tensor = torch.zeros((self.s, self.s, 1))

        p_tensor1 = torch.zeros((self.s, self.s, 1))
        p_ct_tensor1 = torch.zeros((self.s, self.s, 1))
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
            if (branch == 0):
                p_tensor[int(ymax * self.s), int(xmin * self.s)] = 1  # left-bot
                p_tensor1[int(ymax * self.s), int(xmin * self.s)] = 1  # left-bot
            elif (branch == 1):
                p_tensor[int(ymin * self.s), int(xmin * self.s)] = 1  # left-top
                p_tensor1[int(ymin * self.s), int(xmin * self.s)] = 1  # left-top
            elif (branch == 2):
                p_tensor[int(ymax * self.s), int(xmax * self.s)] = 1  # right-bot
                p_tensor1[int(ymax * self.s), int(xmax * self.s)] = 1  # right-bot
            elif (branch == 3):
                p_tensor[int(ymin * self.s), int(xmax * self.s)] = 1  # right-top
                p_tensor1[int(ymin * self.s), int(xmax * self.s)] = 1  # right-top
            p_ct_tensor[int((ymin + ymax) / 2 * self.s), int((xmin + xmax) / 2 * self.s)] = 1  # center
            p_ct_tensor1[int((ymin + ymax) / 2 * self.s), int((xmin + xmax) / 2 * self.s)] = 1  # center

        posi.append(p_tensor)
        posi.append(p_tensor1)
        posi_ct.append(p_ct_tensor)
        posi_ct.append(p_ct_tensor1)

        return posi, posi_ct

    def position_pro(self, branch, boxes) -> torch.tensor:
        ''' get point coordinat
        Args:
            idx: idx-th image
            branch: branch-th result range(0,3)
            boxes: processed boxes
        Returns:
            corner_pos: tensor [2,2,B] (W,H)
            center_pos: tensor [2,2,B]
        '''
        corner_pos = []
        center_pos = []

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
            if (branch == 0):
                corner_pos_temp = torch.tensor([int(xmin * self.s), int(ymax * self.s)])  # left-bot
            elif (branch == 1):
                corner_pos_temp = torch.tensor([int(xmin * self.s), int(ymin * self.s)])  # left-top
            elif (branch == 2):
                corner_pos_temp = torch.tensor([int(xmax * self.s), int(ymax * self.s)])  # right-bot
            elif (branch == 3):
                corner_pos_temp = torch.tensor([int(xmax * self.s), int(ymin * self.s)])  # right-top

            ctx_p = (xmin * self.s + xmax * self.s) / 2
            cty_p = (ymin * self.s + ymax * self.s) / 2
            ctx = int(ctx_p)
            cty = int(cty_p)
            center_pos_temp = torch.tensor([ctx, cty])  # center
            corner_pos.append(corner_pos_temp)
            center_pos.append(center_pos_temp)
        corner_pos = torch.cat(corner_pos, dim=-1)
        center_pos = torch.cat(center_pos, dim=-1)
        return corner_pos, center_pos

    def boxes_tensor_pro(self, boxes) -> torch.tensor:
        ''' process the boxes to result
        Args:
            idx: idx-th image
            boxes: processed box
        Returns:
            boxes: tensor. [self.B,points(4),2,2] precise coordinate of corner
        '''
        boxes_list = []
        points = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            points.append(
                torch.tensor([(center_x * self.s, center_y * self.s), (xmin * self.s, ymax * self.s)]))  # left-bot
            points.append(
                torch.tensor([(center_x * self.s, center_y * self.s), (xmin * self.s, ymin * self.s)]))  # left-top
            points.append(
                torch.tensor([(center_x * self.s, center_y * self.s), (xmax * self.s, ymax * self.s)]))  # right-bot
            points.append(
                torch.tensor([(center_x * self.s, center_y * self.s), (xmax * self.s, ymin * self.s)]))  # right-top
            points_tensor = torch.stack(points)
        if len(boxes) != 0:
            boxes_list.append(points_tensor)
            stacked_tensor = torch.stack(boxes_list).squeeze(0)
            boxes_1 = stacked_tensor.view(int(stacked_tensor.shape[0] / 4), 4, 2, 2)
            # [N,4,2,2]
            return boxes_1
        else:
            return torch.zeros((2, 4, 2, 2))

    def LxLy_tensor_pro(self, ct_pt, corner_pt) -> torch.tensor:
        ''' process the L to result
        Args:
            ct_pt: tensor center point
            corner_pt: tensor corner point
        Returns:
            Lx: tensor [s]
            Ly: tensor [s]
        '''
        Lx_ct = torch.zeros(self.s)
        Ly_ct = torch.zeros(self.s)

        Lx_ct[int(corner_pt[0])] = self.infinite
        Ly_ct[int(corner_pt[1])] = self.infinite

        L_ct = torch.cat((Lx_ct, Ly_ct))

        Lx_cr = torch.zeros(self.s)
        Ly_cr = torch.zeros(self.s)

        Lx_cr[int(ct_pt[0])] = self.infinite
        Ly_cr[int(ct_pt[1])] = self.infinite

        L_cr = torch.cat((Lx_cr, Ly_cr))

        return L_ct, L_cr

    def L_tensor_pro(self, branch, boxes) -> list:
        ''' process the L to result
        Args:
            idx: idx-th image
            branch: branch-th result range(0,3)
            boxes: processed boxes
        Returns:
            Link_ct_list: list_ct includeing ct tensor [self.s,self.s,2*self.s]
            Link_cr_list: list_cr includeing cr tensor [self.s,self.s,2*self.s]
        '''
        Link_ct_list = []
        Link_cr_list = []

        Link_tmp_cr = torch.zeros((self.s, self.s, 2 * self.s))
        Link_tmp_ct = torch.zeros((self.s, self.s, 2 * self.s))

        Link_tmp_cr1 = torch.zeros((self.s, self.s, 2 * self.s))
        Link_tmp_ct1 = torch.zeros((self.s, self.s, 2 * self.s))
        box_tensor = self.boxes_tensor_pro(boxes)
        box_branch = box_tensor[:, branch, ...]  # [self.B,2,2] 感觉应该是[N,2,2]
        for obj_idx, obj_data in enumerate(box_branch):
            L_ct, L_cr = self.LxLy_tensor_pro(obj_data[0], obj_data[1])
            Link_tmp_cr[int(obj_data[1][1]), int(obj_data[1][0])] = L_cr
            Link_tmp_cr1[int(obj_data[1][1]),int(obj_data[1][0]) ] = L_cr
            Link_tmp_ct[int(obj_data[0][1]), int(obj_data[0][0])] = L_ct
            Link_tmp_ct1[int(obj_data[0][1]), int(obj_data[0][0])] = L_ct

        Link_tmp_ct[..., :self.s] = F.softmax(Link_tmp_ct[..., :self.s], dim=-1)
        Link_tmp_ct[..., self.s:] = F.softmax(Link_tmp_ct[..., self.s:], dim=-1)
        Link_tmp_ct1[..., :self.s] = F.softmax(Link_tmp_ct1[..., :self.s], dim=-1)
        Link_tmp_ct1[..., self.s:] = F.softmax(Link_tmp_ct1[..., self.s:], dim=-1)
        Link_tmp_cr[..., :self.s] = F.softmax(Link_tmp_cr[..., :self.s], dim=-1)
        Link_tmp_cr[..., self.s:] = F.softmax(Link_tmp_cr[..., self.s:], dim=-1)
        Link_tmp_cr1[..., :self.s] = F.softmax(Link_tmp_cr1[..., :self.s], dim=-1)
        Link_tmp_cr1[..., self.s:] = F.softmax(Link_tmp_cr1[..., self.s:], dim=-1)
        Link_ct_list.append(Link_tmp_ct)
        Link_ct_list.append(Link_tmp_ct1)

        Link_cr_list.append(Link_tmp_cr)
        Link_cr_list.append(Link_tmp_cr1)
        # 返回lxij+lyij[14,14,2*14]
        return Link_ct_list, Link_cr_list

    def Qij_tensor_pro(self, branch, boxes, labels) -> torch.tensor:
        ''' conbine the classes with position
        Args:
            idx: idx-th image
            branch: branch-th result
            boxes: processed box
            labels: mixed label
        Returns:
            Q_list: list. including [S,S,classes*B] corner point
            Q_ct_list: list. including [S,S,classes*B] ct point
        '''

        Q_list = []
        Q_ct_list = []
        Q_tensor = torch.zeros((self.s, self.s, 20))
        Q_ct_tensor = torch.zeros((self.s, self.s, 20))

        Q_tensor1 = torch.zeros((self.s, self.s, 20))
        Q_ct_tensor1 = torch.zeros((self.s, self.s, 20))
        for idx_ele, item in enumerate(labels):
            # self.boxes[idx][idx_ele] [xmin, ymin, xmax, ymax]
            box = boxes[idx_ele]
            xmin, ymin, xmax, ymax = box
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001


            if (branch == 0):
                Q_tensor[int(ymax * self.s), int(xmin * self.s), item] = self.infinite
                Q_tensor1[int(ymax * self.s), int(xmin * self.s), item ] = self.infinite
            elif (branch == 1):
                Q_tensor[int(ymin * self.s), int(xmin * self.s), item] = self.infinite
                Q_tensor1[int(ymin * self.s), int(xmin * self.s), item] = self.infinite
            elif (branch == 2):
                Q_tensor[int(ymax * self.s), int(xmax * self.s), item ] = self.infinite
                Q_tensor1[int(ymax * self.s), int(xmax * self.s), item] = self.infinite
            elif (branch == 3):
                Q_tensor[int(ymin * self.s), int(xmax * self.s), item] = self.infinite
                Q_tensor1[int(ymin * self.s), int(xmax * self.s), item ] = self.infinite
            Q_ct_tensor[int((ymin + ymax) / 2 * self.s), int((xmin + xmax) / 2 * self.s), item] = self.infinite
            Q_ct_tensor1[int((ymin + ymax) / 2 * self.s), int((xmin + xmax) / 2 * self.s), item] = self.infinite
        Q_ct_tensor = F.softmax(Q_ct_tensor, dim=-1).clone()
        Q_tensor = F.softmax(Q_tensor, dim=-1).clone()
        Q_ct_tensor1 = F.softmax(Q_ct_tensor1, dim=-1).clone()
        Q_tensor1 = F.softmax(Q_tensor1, dim=-1).clone()
        Q_list.append(Q_tensor)
        Q_list.append(Q_tensor1)
        Q_ct_list.append(Q_ct_tensor)
        Q_ct_list.append(Q_ct_tensor1)
        return Q_list, Q_ct_list

    def relative_pos_tensor_pro(self, branch, boxes) -> torch.tensor:
        ''' get relative coordinate to the same grid
        Args:
            idx: idx-th image
            branch: branch-th result range(0,3)
            boxes: processed boxes
        Returns:
            pos_list: list. including [s,s,2*B] corner precise
            pos_ct_list: list. including [s,s,2*B] center precise
        '''
        pos_list = []
        pos_ct_list = []

        pos_tensor = torch.zeros((self.s, self.s, 2))
        pos_ct_tensor = torch.zeros((self.s, self.s, 2))
        pos_tensor1 = torch.zeros((self.s, self.s, 2))
        pos_ct_tensor1 = torch.zeros((self.s, self.s, 2))
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
            if (branch == 0):
                # pos_tensor[int(xmin * self.s), int(ymax * self.s)] = torch.tensor(
                #     [xmin * self.s - int(xmin * self.s), ymax * self.s - int(ymax * self.s)])  # left-top
                pos_tensor[int(ymax * self.s), int(xmin * self.s)] = torch.tensor(
                    [ymax * self.s - int(ymax * self.s), xmin * self.s - int(xmin * self.s)])  # left-top
                # pos_tensor1[int(xmin * self.s), int(ymax * self.s)] = torch.tensor(
                #     [xmin * self.s - int(xmin * self.s), ymax * self.s - int(ymax * self.s)])  # left-top
                pos_tensor1[int(ymax * self.s), int(xmin * self.s)] = torch.tensor(
                    [ymax * self.s - int(ymax * self.s), xmin * self.s - int(xmin * self.s)])
            elif (branch == 1):
                pos_tensor[int(ymin * self.s), int(xmin * self.s)] = torch.tensor(
                    [ymin * self.s - int(ymin * self.s), xmin * self.s - int(xmin * self.s)])  # left-bot
                pos_tensor1[int(ymin * self.s), int(xmin * self.s)] = torch.tensor(
                    [ymin * self.s - int(ymin * self.s), xmin * self.s - int(xmin * self.s)])  # left-bot
            elif (branch == 2):
                pos_tensor[int(ymax * self.s), int(xmax * self.s)] = torch.tensor(
                    [ymax * self.s - int(ymax * self.s), xmax * self.s - int(xmax * self.s)])  # right-top
                pos_tensor1[int(ymax * self.s), int(xmax * self.s)] = torch.tensor(
                    [ymax * self.s - int(ymax * self.s), xmax * self.s - int(xmax * self.s)])  # right-top
            elif (branch == 3):
                pos_tensor[int(ymin * self.s),int(xmax * self.s) ] = torch.tensor(
                    [ymin * self.s - int(ymin * self.s),xmax * self.s - int(xmax * self.s) ])  # right-bot
                pos_tensor1[int(ymin * self.s),int(xmax * self.s) ] = torch.tensor(
                    [ymin * self.s - int(ymin * self.s),xmax * self.s - int(xmax * self.s) ])  # right-bot

            ctx_p = (xmin * self.s + xmax * self.s) / 2
            cty_p = (ymin * self.s + ymax * self.s) / 2
            ctx = int(ctx_p)
            cty = int(cty_p)
            pos_ct_tensor[cty,ctx] = torch.tensor([ctx_p - ctx, cty_p - cty])  # center
            pos_ct_tensor1[cty,ctx] = torch.tensor([ctx_p - ctx, cty_p - cty])  # center

        pos_list.append(pos_tensor)
        pos_list.append(pos_tensor1)
        pos_ct_list.append(pos_ct_tensor)
        pos_ct_list.append(pos_ct_tensor1)
        return pos_list, pos_ct_list

    def combine_list(self, Q_list, Q_ct_list, posi_list, posi_ct_list, Link_ct_list, Link_cr_list, pos_list,
                     pos_ct_list) -> torch.tensor:
        ''' combine the ct with cr points -> [14,14,204]
        Args:
            Q_list: list
            Q_ct_list: list
            posi_list: list
            posi_ct_list: list
            Link_ct_list: list
            Link_cr_list: list
            pos_list: list
            pos_ct_list: list
        Returns:
            final label tensor: [14,14,204]
        '''
        list_feature = []
        list_ct_feature = []
        for i in range(self.B):
            # zip_tensor_ct = torch.cat((Q_ct_list[i], posi_ct_list[i], Link_ct_list[i], pos_ct_list[i]), dim=-1)
            zip_tensor_ct = torch.cat((posi_ct_list[i], pos_ct_list[i],Link_ct_list[i],Q_ct_list[i]), dim=-1)
            # zip_tensor = torch.cat((posi_list[i], pos_list[i],Link_cr_list[i],Q_list[i]), dim=-1)
            zip_tensor = torch.cat((posi_list[i], pos_list[i], Link_cr_list[i], Q_list[i]), dim=-1)
            list_feature.append(zip_tensor)
            list_ct_feature.append(zip_tensor_ct)
        feature_tensor = torch.cat((list_feature), dim=-1)
        feature_ct_tensor = torch.cat((list_ct_feature), dim=-1)
        return torch.cat((feature_ct_tensor, feature_tensor), dim=-1)

    def load_label(self,branch,boxes,labels):
        if len(boxes) != 0:
            # cal Q
            Q_tensor, Q_ct_tensor = self.Qij_tensor_pro(branch,boxes,labels)  # [S,S,classes*B],[S,S,classes*B]
            # cal P,mixed p and note it with lamda
            P_tensor, P_ct_tensor = self.pij_pro(branch, boxes)  # [s,s,1*B],[s,s,1*B]
            # cal L
            Link_ct_list, Link_cr_list = self.L_tensor_pro( branch, boxes)  # [s,s,2*s*B],[s,s,2*s*B]
            # cal x
            x_tensor, x_ct_tensor = self.relative_pos_tensor_pro(branch, boxes)  # [s,s,2*B], [s,s,2*B]

            final_tensor = self.combine_list(Q_tensor, Q_ct_tensor, P_tensor, P_ct_tensor, Link_ct_list, Link_cr_list,
                                             x_tensor, x_ct_tensor)
            # return
            # cr_pt, ct_pt = self.pos_tensor_pro(branch, boxes)
            # final_pt_temp = torch.cat((ct_pt, cr_pt), dim=-1)
            # final_pt = final_pt_temp.reshape(final_pt_temp.shape[0] // 2, 2)
            # return final_tensor, final_pt
            return final_tensor


if __name__ == '__main__':
    device = 'cuda'
    file_root = 'VOCdevkit/VOC2007/JPEGImages/'
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 100
    # 自定义训练数据集
    train_dataset = plnDataset(img_root=file_root, list_file='voctrain.txt', train=True,
                               transform=[transforms.ToTensor()])
    img, target = train_dataset.__getitem__(1)
    print("target encode success")
    #
    # img = img.to("cuda")
    # target = target.to("cuda")
    #
    # model = inceptionresnetv2(num_classes=20, pretrained='imagenet').cuda()
    # # # 加载权重，就是在train.py中训练生成的权重文件yolo.pth
    # model.load_state_dict(torch.load("pln.pth"))
    # # 测试模式
    # model.eval()
    #
    # img = img.unsqueeze(0)
    # print(img.shape)
    # torch.set_printoptions(profile="full")
    # p = model(img)
    # # p = p.permute(0, 2, 3, 1)
    #
    # print(p)
    # print(target.shape)
    # print(target.unsqueeze(0).expand_as(p).shape)
    # target = target.unsqueeze(0).expand_as(p)
    # loss = plnLoss.plnLoss(14, 2, 1, 1, w_link=1)
    # a = loss.forward(p, target)
    # print(a)
