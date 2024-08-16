from plnData import plnDataset
from plnLoss import  plnLoss
from new_resnet import inceptionresnetv2, pretrained_inception
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from new_net import *
import torch


def save_ckpt(epoch):
    # check if the folder exists
    ckpt_save_path = f"pln.pth"
    net.eval()
    print(f"Saving checkpoint to {ckpt_save_path}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_save_path)
    net.train()


def load_ckpt():
    ckpt_path = "pln.pth"
    print(f"Loading checkpoint from {ckpt_path}")
    mode = "train"
    try:
        if mode == 'train':
            checkpoint = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint, resume from epoch {start_epoch}")
        else:
            checkpoint = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)

    except Exception as e:
        print("Failed to load checkpoint")
        raise e

device = 'cuda'
file_root = 'VOCdevkit/VOC2007/JPEGImages/'
batch_size = 2
learning_rate = 0.0001
num_epochs = 1
type = "train"

# 自定义训练数据集
train_dataset = plnDataset(img_root=file_root, list_file='voctrain.txt', train=True, transform=[transforms.ToTensor()])
# 加载自定义的训练数据集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
# 自定义测试数据集
test_dataset = plnDataset(img_root=file_root, list_file='voctest.txt', train=False, transform=[transforms.ToTensor()])
# 加载自定义的测试数据集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
print('the dataset has %d images' % (len(train_dataset)))
net = pretrained_inception()
# 定义优化器  RMS
optimizer = torch.optim.SGD(
    net.parameters(),
    # 学习率
    lr=learning_rate,
    # 动量
    momentum=0.9,
    # 正则化
    weight_decay=5e-4
)

"""
下面这段代码主要适用于迁移学习训练，可以将预训练的ResNet-50模型的参数赋值给新的网络，以加快训练速度和提高准确性。
"""

# net = inceptionresnetv2(num_classes=20, pretrained='imagenet').cuda()

# net = PLN(448)
# # 是否加载之前训练过的模型
# net_static_dict = torch.load("pln.pth")
# net.load_state_dict(net_static_dict)
load_ckpt()

criterion = plnLoss(14,2,w_coord=2,w_link=.5,w_class=.5).to(device)


if type == "train":
    net.train()
    # 开始训练
    for epoch in range(num_epochs):
        # 这个地方形成习惯，因为网络可能会用到Dropout和batchnorm
        net.train()
        # 调整学习率
        if epoch == 60:
            learning_rate = 0.0001
        if epoch == 80:
            learning_rate = 0.00001
        # optimizer.param_groups 返回一个包含优化器参数分组信息的列表，每个分组是一个字典，主要包含以下键值：
        # params：当前参数分组中需要更新的参数列表，如网络的权重，偏置等。
        # lr：当前参数分组的学习率。就是我们要提取更新的
        # momentum：当前参数分组的动量参数。
        # weight_decay：当前参数分组的权重衰减参数。
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate      # 更改全部的学习率
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        # 计算损失
        total_loss = 0.
        # 开始迭代训练
        for i, (images, target) in enumerate(train_loader):
            # print("training",images.shape,target.shape)
            images, target = images.cuda(), target.cuda()
            pred = net(images)
            # target torch.Size([batch, 4, 14, 14, 204])
            # pred torch.Size([4, batch, 204, 14, 14])
            target = target.permute(1, 0 ,2, 3, 4)
            # 创建损失函数
            # [batch_size,4,14,14,204]
            # loss = criterion(pred,target)
            batch_size = pred[0].shape[0]
            loss0 = criterion(pred[0], target[0])
            loss1 = criterion(pred[1], target[1])
            loss2 = criterion(pred[2], target[2])
            loss3 = criterion(pred[3], target[3])
            loss = (loss0 + loss1 + loss2 + loss3)/batch_size
            total_loss += loss.item()

            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数优化
            optimizer.step()
            if (i + 1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch +1, num_epochs, i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
        # 开始测试
        validation_loss = 0.0
        net.eval()
        for i, (images, target) in enumerate(test_loader):
            images, target = images.cuda(), target.cuda()
            # 输入图像
            pred = net(images)
            target = target.permute(1, 0, 2, 3, 4)

            # 计算损失
            batch_size = pred[0].shape[0]
            loss0 = criterion(pred[0], target[0])
            loss1 = criterion(pred[1], target[1])
            loss2 = criterion(pred[2], target[2])
            loss3 = criterion(pred[3], target[3])
            loss = (loss0 + loss1 + loss2 + loss3) / batch_size
            total_loss += loss.item()
            # 累加损失
            validation_loss += loss.item()
            # 计算平均loss
        validation_loss /= len(test_loader)

        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        # 保存模型参数
        save_ckpt(epoch=num_epochs)
        # torch.save(net.state_dict(), 'pln.pth')




