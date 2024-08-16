from collections import Counter

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

from learn.PLN2.predict import model

score_confident = 0.15
p_confident = 0.95
nms_confident = 0.15
# 设置iou阈值
iou_con = 0.5

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
CLASS_NUM = 20


class Pred1():
    # 参数初始化
    def __init__(self, model):
        self.model = model

    def compute_area(self, branch, j, i):
        area = [[], []]
        if branch == 0:
            # 左下角
            area = [[0, j + 1], [i, 14]]
        elif branch == 1:
            # 左上角
            area = [[0, j + 1], [0, i + 1]]
        elif branch == 2:
            # 右下角
            area = [[j, 14], [i, 14]]
        elif branch == 3:
            # 右上角
            area = [[j, 14], [0, i + 1]]
        #
        # x_area = [[0, a], [0, a],  [a+1, 14],[a+1, 14]]
        # y_area = [[b+1, 14], [0, b], [b+1, 14],[0, b]]
        return area

    def result(self, img_root):
        # 读取测试的图像
        img = cv2.imread(img_root)
        # 获取高宽信息
        h, w, _ = img.shape
        # 调整图像大小
        image = cv2.resize(img, (448, 448))
        # CV2读取的图像是BGR，这里转回RGB模式
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 图像均值
        mean = (123, 117, 104)  # RGB
        # 减去均值进行标准化操作
        img = img - np.array(mean, dtype=np.float32)
        # 创建数据增强函数
        transform = ToTensor()
        # 图像转为tensor，因为cv2读取的图像是numpy格式
        img = transform(img)
        # 输入要求是BCHW，加一个batch维度
        img = img.unsqueeze(0)
        img = img.to("cuda")
        # print("img",img.shape)
        # 图像输入模型，返回值为1*7*7*204的张量

        Result = self.model(img)
        # Result torch.Size([1, 204, 14, 14])
        # result torch.Size([14, 14, 204])
        # 获取目标的边框信息

        Result0 = Result[0].permute(0, 2, 3, 1).squeeze(0)
        bbox0 = self.Decode(0, Result0)
        Result1 = Result[1].permute(0, 2, 3, 1).squeeze(0)
        bbox1 = self.Decode(1, Result1.clone())
        Result2 = Result[2].permute(0, 2, 3, 1).squeeze(0)
        bbox2 = self.Decode(2, Result2.clone())
        Result3 = Result[3].permute(0, 2, 3, 1).squeeze(0)
        bbox3 = self.Decode(3, Result3)
        # print("bbox0", bbox0.shape)
        # print("bbox1", bbox1.shape)
        # print("bbox2", bbox2.shape)
        # print("bbox3", bbox3.shape)
        bbox = torch.cat((bbox0, bbox1, bbox2, bbox3), dim=0)
        # print("bbox", bbox.shape)
        # print(bbox)
        # 非极大值抑制处理
        bboxes = self.NMS(bbox)  # n*6   bbox坐标是基于7*7网格需要将其转换成448
        if len(bboxes) == 0:
            print("未识别到任何物体")
            print("尝试减小 confident 以及 iou_con")
            print("也可能是由于训练不充分，可在训练时将epoch增大")
        for i in range(0, len(bboxes)):  # bbox坐标将其转换为原图像的分辨率
            x1 = bboxes[i][0].item()  # 后面加item()是因为画框时输入的数据不可一味tensor类型
            y1 = bboxes[i][1].item()
            x2 = bboxes[i][2].item()
            y2 = bboxes[i][3].item()
            score = bboxes[i][4].item()
            class_name = bboxes[i][5].item()
            print(x1, y1, x2, y2, VOC_CLASSES[int(class_name)], bboxes[i][4].item())
            text = VOC_CLASSES[int(class_name)] + '{:.3f}'.format(score)
            print("text:", text)
            # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 画框
            # cv2.putText(image, text, (int(x1) + 10, int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            # plt.imsave('test_001.jpg', image)
        return bboxes

    # 接受的result的形状为1*7*7*204
    def Decode(self, branch, result):
        result = result.squeeze()
        # result [14*14*204]
        # 0 pij 1x 2y
        # 3-16 lxij 16-30 lyij
        # 31-50 qij
        r = []
        bboxes_ = list()
        labels_ = list()
        scores_ = list()
        for i in range(14):
            for j in range(14):
                for p in range(4):
                    result[i, j, 51 * p + 3:51 * p + 17] = torch.softmax(result[i, j, 51 * p + 3:51 * p + 17], dim=-1)
                    result[i, j, 51 * p + 17:51 * p + 31] = torch.softmax(result[i, j, 51 * p + 17:51 * p + 31], dim=-1)
                    result[i, j, 51 * p + 31:51 * p + 51] = torch.softmax(result[i, j, 51 * p + 31:51 * p + 51], dim=-1)

        for p in range(2):
            # ij center || mn corner
            for i in range(14):
                for j in range(14):
                    # print("start,", i, j)
                    if result[i, j, p * 51] < p_confident:
                        continue
                    # print("pij", p, result[i, j, p * 51])
                    # 右上角启发区域
                    # x_area, y_area = [j, 14], [0, i + 1]
                    x_area, y_area = self.compute_area(branch, j, i)
                    for n in range(y_area[0], y_area[1]):
                        for m in range(x_area[0], x_area[1]):
                            for c in range(20):
                                p_ij = result[i, j, 51 * p + 0]
                                p_nm = result[n, m, 51 * (p + 2) + 0]
                                i_, j_, n_, m_ = result[i, j, 51 * p + 2], result[i, j, 51 * p + 1], result[
                                    n, m, 51 * (p + 2) + 2], result[n, m, 51 * (p + 2) + 1]
                                l_ij_x = result[i, j, 51 * p + 3 + m]
                                l_ij_y = result[i, j, 51 * p + 3 + n]
                                l_nm_x = result[n, m, 51 * (p + 2) + 17 + j]
                                l_nm_y = result[n, m, 51 * (p + 2) + 17 + i]
                                q_cij = result[i, j, 51 * p + 31 + c]
                                q_cnm = result[n, m, 51 * (p + 2) + 31 + c]
                                score = p_ij * p_nm * q_cij * q_cnm * (l_ij_x * l_ij_y + l_nm_x * l_nm_y) / 2
                                score *= 1000
                                # print("p_ij:",p_ij,"p_nm:",p_nm,"l_ij_x:",l_ij_x,"l_ij_y:",l_ij_x,"l_nm_x:","l_nm_y:",l_nm_y)

                                # print(p_ij,p_nm,l_ij_x,l_ij_y,l_nm_x,l_nm_y,q_cij,q_cnm)
                                # print(score)
                                # 设置score阈值
                                # if score>0:
                                #     print(i, j, n, m)
                                #     print("score", score)
                                if score > score_confident:
                                #     print(i, j, n, m)
                                #     print("score", score)
                                    r.append([i + i_, j + j_, n + n_, m + m_, c, score])
            for l in r:
                # 重新encode 变为xmin,ymin,xmax,ymax,score.class
                if branch == 0:
                    # 左下角
                    bbox = [l[3], 2 * l[0] - l[2], 2 * l[1] - l[3], l[2]]
                elif branch == 1:
                    # 左上角
                    bbox = [l[3], l[2], 2 * l[1] - l[3], 2 * l[0] - l[2]]
                elif branch == 2:
                    # 右下角
                    bbox = [2 * l[1] - l[3], 2 * l[0] - l[2], l[3], l[2]]
                elif branch == 3:
                    # 右上角
                    bbox = [2 * l[1] - l[3], l[2], l[3], 2 * l[0] - l[2]]

                # print(bbox)
                bbox = [b * 32 for b in bbox]
                bboxes_.append(bbox)
                labels_.append(l[4])
                scores_.append(l[5])  # result of a img
                # bboxes_nms = self._suppress(bboxes_, scores_)
        # print(bboxes_)
        # print(labels_)
        # print(scores_)
        bbox_info = torch.zeros(len(labels_), 6)
        for i in range(len(labels_)):
            bbox_info[i, 0] = bboxes_[i][0]
            bbox_info[i, 1] = bboxes_[i][1]
            bbox_info[i, 2] = bboxes_[i][2]
            bbox_info[i, 3] = bboxes_[i][3]
            bbox_info[i, 4] = scores_[i]
            bbox_info[i, 5] = labels_[i]

        # print("bbox_info success")
        # print(bbox_info)
        return bbox_info

    # 非极大值抑制处理，按照类别处理，bbox为Decode获取的预测框的位置信息和类别概率和类别信息
    def NMS(self, bbox, iou_con=iou_con):
        # 存放最终需要保留的预测框
        bboxes = []
        # 取出每个gird cell中的类别信息，返回一个列表
        ori_class_index = bbox[:, 5]
        # 按照类别进行排序，从高到低，返回的是排序后的类别列表和对应的索引位置,如下：
        """
        类别排序
        tensor([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
         3.,  3.,  3.,  4.,  4.,  4.,  4.,  5.,  5.,  5.,  6.,  6.,  6.,  6.,
         6.,  6.,  6.,  6.,  7.,  8.,  8.,  8.,  8.,  8., 14., 14., 14., 14.,
        14., 14., 14., 15., 15., 16., 17.], grad_fn=<SortBackward0>)
        位置索引
        tensor([48, 47, 46, 45, 44, 43, 42,  7,  8, 22, 11, 16, 14, 15, 24, 20,  1,  2,
         6,  0, 13, 23, 25, 27, 32, 39, 38, 35, 33, 31, 30, 28,  3, 26, 10, 19,
         9, 12, 29, 41, 40, 21, 37, 36, 34, 18, 17,  5,  4])
        """
        class_index, class_order = ori_class_index.sort(dim=0, descending=False)
        # class_index是一个tensor，这里把他转为列表形式
        class_index = class_index.tolist()
        # 根据排序后的索引更改bbox排列顺序
        bbox = bbox[class_order, :]
        a = 0
        for i in range(0, CLASS_NUM):
            # 统计目标数量，即某个类别出现在grid cell中的次数
            num = class_index.count(i)
            # 预测框中没有这个类别就直接跳过
            if num == 0:
                continue
            # 提取同一类别的所有信息
            x = bbox[a:a + num, :]
            # 提取真实类别概率信息
            score = x[:, 4]
            # 提取出来的某一类别按照真实类别概率信息高度排序，递减
            score_index, score_order = score.sort(dim=0, descending=True)
            # 根据排序后的结果更改真实类别的概率排布
            y = x[score_order, :]
            # 先看排在第一位的物体的概率是否大有给定的阈值，不满足就不看这个类别了，丢弃全部的预测框
            if y[0, 4] >= nms_confident:
                for k in range(0, num):
                    # 真实类别概率，排序后的
                    y_score = y[:, 4]
                    # 对真实类别概率重新排序，保证排列顺序依照递减，其实跟上面一样的，多此一举
                    _, y_score_order = y_score.sort(dim=0, descending=True)
                    y = y[y_score_order, :]
                    # 判断概率是否大于0
                    if y[k, 4] > 0:
                        # 计算预测框的面积
                        area0 = (y[k, 2] - y[k, 0]) * (y[k, 3] - y[k, 1])
                        if area0 < 200:
                            y[k, 4] = 0
                            continue
                        for j in range(k + 1, num):
                            # 计算剩余的预测框的面积
                            area1 = (y[j, 2] - y[j, 0]) * (y[j, 3] - y[j, 1])
                            if area1 < 200:
                                y[j, 4] = 0
                                continue
                            x1 = max(y[k, 0], y[j, 0])
                            x2 = min(y[k, 2], y[j, 2])
                            y1 = max(y[k, 1], y[j, 1])
                            y2 = min(y[k, 3], y[j, 3])
                            w = x2 - x1
                            h = y2 - y1
                            if w < 0 or h < 0:
                                w = 0
                                h = 0
                            inter = w * h
                            # 计算与真实目标概率最大的那个框的iou
                            iou = inter / (area0 + area1 - inter)
                            # iou大于一定值则认为两个bbox识别了同一物体删除置信度较小的bbox
                            # 同时物体类别概率小于一定值也认为不包含物体
                            if iou >= iou_con or y[j, 4] < nms_confident:
                                y[j, 4] = 0
                for mask in range(0, num):
                    if y[mask, 4] > 0:
                        bboxes.append(y[mask])
            # 进入下个类别
            a = num + a
        #     返回最终预测的框
        return bboxes


def get_pred(fname):
    img = cv2.imread(fname)
    h, w, _ = img.shape
    bboxes = Pred1.result(fname)
    print(bboxes)
    for i in range(len(bboxes)):
        # xmin ymin xmax ymax class confident
        # 204.7745361328125 106.17108154296875 433.3826904296875 305.6472473144531 cat 0.17054273188114166
        bboxes[i][0] = bboxes[i][0] / 448 * w
        bboxes[i][1] = bboxes[i][1] / 448 * h
        bboxes[i][2] = bboxes[i][2] / 448 * w
        bboxes[i][3] = bboxes[i][3] / 448 * h
    return bboxes


def get_all_label():
    fnames = []
    # 位置信息
    boxes = []
    # 类别信息
    labels = []
    file_txt = open("voceval.txt")
    objects = []

    # 读取txt文件每一行
    lines = file_txt.readlines()
    # 逐行开始操作
    for line in lines:
        # 去除字符串开头和结尾的空白字符，然后按照空白字符（包括空格、制表符、换行符等）分割字符串并返回一个列表
        splited = line.strip().split()
        # 存储图片的名字
        fnames.append(splited[0])
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
            # 存储位置信息[xmin,ymin,xmax,ymax]
            box.append([x, y, x2, y2])
            # 存储标签信息
            label.append(int(c))
        obj = {}
        obj["bbox"] = box
        obj["label"] = label
        obj["fname"] = splited[0]
        objects.append(obj)
        # 解析完所有行的信息后把所有的位置信息放到boxes列表中，boxes里面的是每一张图的坐标信息，也是一个个列表，即形式是[[[x1,y1,x2,y2],[x3,y3,x4,y4]],[[x5,y5,x5,y6]]...]这样的
        boxes.append(torch.Tensor(box))
    return objects


def get_label(bbox, fname):
    for i in bbox:
        if i["fname"] == fname:
            return i


def draw_target(image, target_boxes, target_labels):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(0, len(target_boxes)):  # bbox坐标将其转换为原图像的分辨率
        x1 = target_boxes[i][0]
        y1 = target_boxes[i][1]
        x2 = target_boxes[i][2]
        y2 = target_boxes[i][3]
        class_name = target_labels[i]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # 画框
        plt.imsave('test_001.jpg', image)


def draw_pred(image, pred_boxes):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(0, len(pred_boxes)):  # bbox坐标将其转换为原图像的分辨率
        x1 = pred_boxes[i][0].item()  # 后面加item()是因为画框时输入的数据不可一味tensor类型
        y1 = pred_boxes[i][1].item()
        x2 = pred_boxes[i][2].item()
        y2 = pred_boxes[i][3].item()
        score = pred_boxes[i][4].item()
        class_name = pred_boxes[i][5].item()
        print(x1, y1, x2, y2, VOC_CLASSES[int(class_name)], pred_boxes[i][4].item())
        text = VOC_CLASSES[int(class_name)] + '{:.3f}'.format(score)
        print("text:", text)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 画框
        cv2.putText(image, text, (int(x1) + 10, int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        plt.imsave('test_001.jpg', image)

def insert_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]  # shape:[N,1]

    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # 计算交集区域面积
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def mean_average_precision(pred_bboxes, true_boxes, iou_threshold, num_classes=20):
    # pred_bboxes(list): [[train_idx,class_pred,prob_score,x1,y1,x2,y2], ...]
    # pred_bboxes 代表所有预测框，true_boxes代表所有真实框，
    # iou_threshold代表设定的IoU阈值，num_classes是总类别数。

    average_precisions = []  # 存储每一个类别的AP
    epsilon = 1e-6  # 防止分母为0
    # 对于每一个类别
    for c in range(num_classes):
        detections = []  # 存储预测为该类别的bbox
        ground_truths = []  # 存储本身就是该类别的bbox(GT)
        # 将预测为该类别的框存储在detections列表中，将本身就是该类别的真实框存储在ground_truths列表中
        for detection in pred_bboxes:
            if detection[1] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # print(detections)
        # print(ground_truths)

        # 统计每一张图片中真实框的个数，gt[0]是train_idx，它指示了用于区分每张图片的一个编号。
        # img0 3 bboxes img1 5 bboxes
        # amount_bboxes={0:3,1:5}
        amount_bboxes = Counter(gt[0] for gt in ground_truths)
        for key, val in amount_bboxes.items():
            # amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}
            # amout_bboxes包含了每张图片中真实框的个数，它是一个字典，其中key为图片编号，value为该图片包含的真实框的个数；
            # 之后，将真实框的个数用全0向量来替代，有几个真实框，全0向量就包含几个0。
            amount_bboxes[key] = torch.zeros(val)  # 置0，表示这些真实框初始时都没有与任何预测框匹配

        # print(amount_bboxes)

        # 将预测框按照置信度从大到小排序
        detections.sort(key=lambda x: x[2], reverse=True)
        # print(detections)
        # 初始化TP,FP
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        # TP+FN就是当前类别GT框的总数，是固定的
        total_true_bboxes = len(ground_truths)
        # print("total_true_bboxes",total_true_bboxes)
        # 如果当前类别一个GT框都没有，那么直接跳过即可
        if total_true_bboxes == 0:
            continue

        # 对于每个预测框，先找到它所在图片中的所有真实框，然后计算预测框与每一个真实框之间的IoU，
        # 大于IoU阈值且该真实框没有与其他预测框匹配，则置该预测框的预测结果为TP，否则为FP
        for detection_idx, detection in enumerate(detections):
            # 在计算IoU时，只能是同一张图片内的框做，不同图片之间不能做
            # 图片的编号存在第0个维度
            # 于是下面这句代码的作用是：找到当前预测框detection所在图片中的所有真实框，用于计算IoU
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            # print("ground_truth_img",ground_truth_img)
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                # 计算当前预测框detection与它所在图片内的每一个真实框的IoU
                iou = insert_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_threshold:
                # 这里的detection[0]是amount_bboxes的一个key，best_gt_idx是该key对应的真实框中的train_idx
                if amount_bboxes[detection[0]][
                    best_gt_idx] == 0:  # 只有没被占用的真实框才能用，0表示未被占用（占用：该真实框与某预测框匹配【两者IoU大于设定的IoU阈值】）
                    TP[detection_idx] = 1  # 该预测框为TP
                    amount_bboxes[detection[0]][
                        best_gt_idx] = 1  # 将该真实框标记为已经用过了，不能再用于其他预测框。因为一个预测框最多只能对应一个真实框（最多：IoU小于IoU阈值时，预测框没有对应的真实框)
                else:
                    FP[detection_idx] = 1  # 虽然该预测框与真实框中的一个框之间的IoU大于IoU阈值，但是这个真实框已经与其他预测框匹配，因此该预测框为FP
            else:
                FP[detection_idx] = 1  # 该预测框与真实框中的每一个框之间的IoU都小于IoU阈值，因此该预测框直接为FP

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # 套公式
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        # 把[0,1]这个点加入其中
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # 使用trapz计算AP
        average_precisions.append(torch.trapz(precisions, recalls))
        print(VOC_CLASSES[int(c)])
        print(torch.trapz(precisions, recalls))
    return sum(average_precisions) / len(average_precisions)


model.eval()
Pred1 = Pred1(model)
# fname = "000019.jpg"
# pred = get_pred("../VOCdevkit/VOC2007/JPEGImages/" + fname)


# bboxes = get_all_label()
# target = get_label(bboxes,fname)
# print("target:", target)


# # 蓝色为预测框
# image = cv2.imread("../VOCdevkit/VOC2007/JPEGImages/"+fname)
# draw_pred(image,pred)
# # 红色为标签框
# image  = cv2.imread("test_001.jpg")
# draw_target(image,target["bbox"],target["label"])

bboxes = get_all_label()
# 构建标签
true_boxes = []
for i in bboxes:
    for j in range(len(i["bbox"])):
        box = ["",0,0,0,0,0,0]
        box[0] = i["fname"]
        box[1] = i["label"][j]
        box[2] = 1
        box[3],box[4],box[5],box[6] = i["bbox"][j]
        true_boxes.append(box)


file_txt = open("pred.txt")
pred_bboxes = []
# 读取txt文件每一行
lines = file_txt.readlines()
for line in lines:
    # 去除字符串开头和结尾的空白字符，然后按照空白字符（包括空格、制表符、换行符等）分割字符串并返回一个列表
    splited = line.strip().split()
    box = splited
    for i in range(1,7):
        box[i] = float(box[i])
    pred_bboxes.append(box)
print("pred",pred_bboxes)
print("true",true_boxes)
mean_average_precision(pred_bboxes,true_boxes,0.5)

# file_txt = open("voceval.txt")
# objects = []
# # 读取txt文件每一行
# lines = file_txt.readlines()
# # 逐行开始操作
# fname_list = []
# for line in lines:
#     # 去除字符串开头和结尾的空白字符，然后按照空白字符（包括空格、制表符、换行符等）分割字符串并返回一个列表
#     splited = line.strip().split()
#     fname_list.append(splited[0])
# print(fname_list)
#
# train_set = open('pred.txt', 'w')
# pred_bboxes = []
# for fname in fname_list:
#     image = get_pred("../VOCdevkit/VOC2007/JPEGImages/"+fname)
#     pred = get_pred("../VOCdevkit/VOC2007/JPEGImages/"+fname)
#     for i in pred:
#         box = ["",0,0,0,0,0,0]
#         box[0] = fname
#         box[1] = i[5].tolist()
#         box[2] = i[4].tolist()
#         box[3],box[4],box[5],box[6] = i[:4].tolist()
#         pred_bboxes.append(box)
#         train_set.write(box[0]+" "+ str(box[1])+" "+str(box[2])+" "+ str(box[3])+" "+str(box[4])+" "+ str(box[5])+" "+str(box[6])+"\n")
# print(pred_bboxes)



