import numpy as npimport torch
#导入内置cifar
from torchvision.datasets import cifar
#预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#Compose将一些转换函数组合在一起#ToTensor，原始数据是numpy，现在改成Tensor。会将数据从[0,255]归一化到[0,1]，即除以255
transforms=transforms.Compose([transforms.ToTensor()])
trainData=cifar.CIFAR10('./picdata',train=True,transform=transforms,download=True)
testData=cifar.CIFAR10('./picdata',train=False,transform=transforms)
x=0
for images, labels in trainData:    
    plt.subplot(3,3,x+1)    plt.tight_layout()    images = images.numpy().transpose(1, 2, 0)  # 把channel那一维放到最后    plt.title(str(classes[labels]))    plt.imshow(images)    plt.xticks([])    plt.yticks([])    x+=1    if x==9:        breakplt.show()


import numpy as np
import torch
#导入内置cifar
from torchvision.datasets import cifar
#预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#Compose将一些转换函数组合在一起
#ToTensor，原始数据是numpy，现在改成Tensor。会将数据从[0,255]归一化到[0,1] 除以255
#Normalize则是将数据按照通道进行标准化，(输入[通道]-均值[通道])/标准差[通道]，将数据归一化到[-1,1]
#如果数据在[0,1]之间，则实际的偏移量bias会很大。而一般模型初始化的时候，bias=0，这样收敛的就会慢。经过Normalize后加快收敛速度
#后面两个0.5就是制定mean和std，原来[0,1]变成：(0-0.5)/0.5=-1，(1-0.5)/0.5=1。本例是要灰度化，就一个通道，如果是三通道RGB，则应该为[0.5,0.5,0.5]  ,
transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
trainData=cifar.CIFAR10('./picdata',train=True,transform=transforms,download=True)
testData=cifar.CIFAR10('./picdata',train=False,transform=transforms)
#shuffle随机打乱
trainLoader=DataLoader(trainData,batch_size=64,shuffle=False)
testLoader=DataLoader(testData,batch_size=128,shuffle=False)
#enumerate组合成一个索引序列，同时列出数据下标和数据
examples=enumerate(trainLoader)
batchIndex,(imgData,labels)=next(examples)
fig=plt.figure()
for i in range(9):    
    plt.subplot(3,3,i+1)    
    plt.tight_layout()    
    plt.imshow(imgData[i][0],cmap='gray',interpolation='none')    
    plt.title("{}".format(classes[labels[i]]))    
    plt.xticks([])    
    plt.yticks([])
plt.show()