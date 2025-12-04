#  **Experiment6--InceptionV3_Fashion-MINST** (实验6-InceptionV3实现Fashion-MNIST分类)
##### The experiment is based on the InceptionV3 architecture, conducting classification training and testing  on a portion of  the Fashion-MNIST dataset to verify the feature extraction effect of multi-branch convolution modules. 
###### 本实验基于 InceptionV3 架构，对部分 Fashion-MNIST 数据集（转 3 通道、缩至 299×299）进行分类训练测试，验证了多分支卷积模块的特征提取效果。 
##

## 1.Experimental Purpose
##### 1.Master the basic principles of the convolutional neural network based on the InceptionV3 architecture.
##### 2.Conduct classification training and testing on the Fashion-MNIST dataset using the InceptionV3 architecture.
##### 3.Learn to load pre-trained weights and analyze the network architecture.

###### 1.掌握InceptionV3架构卷积神经网络基本原理。
###### 2.利用InceptionV3架构对Fashion-MNIST数据集进行分类训练和测试。
###### 3.学习调用预训练权重，并分析网络架构。

##

## 2.Experimental Content
##### Due to computational resource constraints, the data volume of the Fashion-MNIST dataset is limited, and the image size is converted to adapt to the InceptionV3 network architecture (3×299×299). Load the pre-trained weights of InceptionV3, and adjust the final fully connected layer to perform a classification task with 10 outputs. Finally, conduct a detailed analysis of the InceptionV3 network architecture model.
###### 由于计算资源的限制，对数据集Fashion-MNIST数据量进行限制，对图像的尺寸进行转换为适应InceptionV3网络架构（3x299x299）。调用InceptionV3预训练权重，并调整最后一层全连接层做输出为10的分类任务。最后对InceptionV3网络架构的模型具体分析。
##
#### 2.1.Fashion-MNIST Dataset Loading and Processing
##
##### （1）Import Pytorch and related tool libraries
###### 导入 PyTorch 及相关工具库
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import Subset
from torchsummary import summary
import numpy as np
```
##
##### （2）Set the training batch size and runtime device, load the dataset, and perform image size conversion and tensor format transformation（3x299x299）on the data images. Set a scaling factor n to control the data volume, and after setting a random seed, randomly select a certain number of datasets to serve as the training set and test set.
###### 设置训练批次大小和运行设备，并加载数据集，对数据图片进行尺寸大小的转换（3x299x299）以及张量格式的变化。设置比例系数n控制数据量，并设置随机种子后，随机抽选一定数量的数据集作为训练集和测试集。
```
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    # 将图像缩放到299×299尺寸（InceptionV3要求的输入尺寸）
    transforms.Resize(299),
    # 将FashionMNIST的单通道灰度图转换为3通道灰度图（InceptionV3要求3通道输入，三通道值相同）
    transforms.Grayscale(num_output_channels=3),
    # 将PIL图像/NumPy数组转换为PyTorch张量，同时将像素值从[0,255]归一化到[0.0,1.0]
    transforms.ToTensor()
])
# 加载FashionMNIST完整训练集，自动下载至data目录
train_full = datasets.FashionMNIST('data',train=True,download=True,transform=transform)
test_full = datasets.FashionMNIST('data',train=False,download=True,transform=transform)
```
##
###### 选取不同比例数据,创建一个固定随机种子的numpy随机数生成器,随机抽取子集索引
```
n = 10 
rng = np.random.default_rng(42)

# 从训练集全量数据的索引中随机抽取子集索引
# replace=False：不重复抽样（保证每个索引只选一次，避免同一个样本被多次选中）
train_idx = rng.choice(len(train_full), len(train_full)//n, replace=False)    # train_idx是一个一维数组，存放被选中的训练集样本索引
test_idx = rng.choice(len(test_full), len(test_full)//n, replace=False)
```
##
###### PyTorch核心数据加载器：将数据集封装为可迭代的批次迭代器
```
train_loader = torch.utils.data.DataLoader(Subset(train_full,train_idx),# 从完整数据集中截取指定子集
                                           batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(Subset(test_full,test_idx),
                                           batch_size=batch_size,shuffle=True)
```
##
#### 2.2.Construct and adjust the network architecture
##### Load the pre-trained weights of InceptionV3, and adjust the final fully connected layer to perform a classification task with 10 outputs.
###### 调用InceptionV3预训练权重，并调整最后一层全连接层做输出为10的分类任务。
```
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features,10)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-4)
```
##
#### 2.3.Model Training and Evaluation
##### Configure the number of training epochs and demonstrate the procedure, including the forward pass, loss computation, and backpropagation.
###### 设置训练轮数并且展示数据训练过程的前向传播，损失计算，反向传播等流程。
###### Training Process
```
epochs=10
accs,losses=[],[]
for epoch in range(epochs):
    for batch_idx,(x,y) in enumerate(trainloader):
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        out = model(x)
        out = outputs.logits  # 主分类输出
        loss = F.cross_entropy(out,y)
        loss.backward()
        optimizer.step()
```
###### Testing Process
```
with torch.no_grad():
        for batch_idx,(x,y) in enumerate(testloader):
            x,y = x.to(device),y.to(device)
            out = model(x)
            total_loss +=F.cross_entropy(out,y).item()
            correct +=(out.argmax(1)==y).sum().item()
```
##
#### 2.4.Introduction to the Network Architecture(网络架构介绍)
##
##### (1)Results of the dataset
```
train_full(下载好的数据集的格式)
```
##
```
Dataset FashionMNIST
    Number of datapoints: 60000
    Root location: data
    Split: Train
    StandardTransform
Transform: Compose(
               Resize(size=299, interpolation=bilinear, max_size=None, antialias=True)
               Grayscale(num_output_channels=3)
               ToTensor()
           )
```
##
```
train_loader(随机选取完的训练集格式)
```
##
```
<torch.utils.data.dataloader.DataLoader at 0x15963dcea40>
```
##
##### (2)Introduction to the Output of the InceptionV3 Model(InceptionV3的model输出介绍)
> 输入图片（299x299x3）
> ↓ 浅层卷积+池化（Conv2d_1a,maxpool2）
> ↓ InceptionA（Mixed_5b/5c/5d）
> ↓ InceptionB（Mixed_6a）
> ↓ InceptionC（Mixed_6b/6c/6d/6e）
> ↓ InceptionD（Mixed_7a）
> ↓ InceptionE（Mixed_7b/7c）
> ↓ 池化+全连接 → 输出10分类结果
```
model
```
##
```
Inception3(
  (Conv2d_1a_3x3): BasicConv2d(
    (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )
  (Conv2d_2a_3x3): BasicConv2d(
    (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )

  ......
  ......

  (Mixed_5c): InceptionA(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
   ......
   ......
  )
  (Mixed_5d): InceptionA(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
   ......
   ......
  )
  (Mixed_6a): InceptionB(
    (branch3x3): BasicConv2d(
      (conv): Conv2d(288, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    ......
    ......
  )
  (Mixed_6b): InceptionC(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    ......
    ......
  )
  (Mixed_6c): InceptionC(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
     ......
     ......
  )
  (Mixed_6d): InceptionC(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
   ......
   ......
  )
  (Mixed_6e): InceptionC(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
   .......
   .......
  )
  (AuxLogits): InceptionAux(
    (conv0): BasicConv2d(
      (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
   ......
   ......
  )
  (Mixed_7b): InceptionE(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    ......
    ......
  )
  (Mixed_7c): InceptionE(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(2048, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_1): BasicConv2d(
      (conv): Conv2d(2048, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )

  ......
  ......

 )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (dropout): Dropout(p=0.5, inplace=False)
  (fc): Linear(in_features=2048, out_features=10, bias=True)
```
