#  **Experiment6--InceptionV3_Fashion-MINST** (实验6-InceptionV3实现Fashion-MNIST分类)
##### The experiment is based on the InceptionV3 architecture, conducting classification training and testing  on a portion of  the Fashion-MNIST dataset to verify the feature extraction effect of multi-branch convolution modules. 
###### 本实验基于 InceptionV3 架构，对部分 Fashion-MNIST 数据集（转 3 通道、缩至 299×299）进行分类训练测试，验证了多分支卷积模块的特征提取效果。 
##

## 1.Exprimental Purpose
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
#### 2.2.Model Training and Evaluation
##### Configure the number of training epochs and demonstrate the procedure, including the forward pass, loss computation, and backpropagation.
###### 设置训练轮数并且展示数据训练过程的前向传播，损失计算，反向传播等流程。
###### Training Process
```
epochs=30
accs,losses=[],[]
for epoch in range(epochs):
    for batch_idx,(x,y) in enumerate(trainloader):
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out,y)
        loss.backward()
        optimizer.step()
```
###### Testing Process
```
with torch.no_grad():
        for batch_idx,(x,y) in enumerate(testloader):
            x,y=x.to(device),y.to(device)
            out=model(x)
            testloss +=F.cross_entropy(out,y).item()
            pred=out.max(dim=1,keepdim=True)[1]
            correct +=pred.eq(y.view_as(pred)).sum().item()
```

##
#### 2.3.Feature Map Visualization
##### Extract and visualize feature maps from convolutional layers.
###### 提取并绘制卷积层的输出特征图。
```
feature1=F.sigmoid(model.conv1(x))
feature2=F.sigmoid(model.conv2(feature1))
n=5
img = x.detach().cpu().numpy()[:n]
feature_map1=feature1.detach().cpu().numpy()[:n]
feature_map2=feature2.detach().cpu().numpy()[:n]

fig,ax=plt.subplots(3,n,figsize=(10,10))
for i in range(n):
    ax[0,i].imshow(img[i].sum(0),cmap='gray')
    ax[1,i].imshow(feature_map1[i].sum(0),cmap='gray')
    ax[2,i].imshow(feature_map2[i].sum(0),cmap='gray')
plt.show()
```
###### 代码中的一些参数解释：
###### feature_map2=feature2.detach().cpu().numpy()[:n]
###### x.detach() - 从计算图中分离，不跟踪梯度
###### .cpu() - 从GPU移到CPU（如果用了GPU）
###### .numpy() - 转成numpy数组，便于matplotlib显示
###### [:n] - 只取前n个样本
###### 结果：原始输入图像的numpy数组

##
