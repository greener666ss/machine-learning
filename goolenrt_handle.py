import torch.nn as nn
from torch.nn import Sequential
import torchvision
from torchvision import transforms
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class my_gooletnet_block(nn.Module):
    def __init__(self,inchannel, c1, c2, c3, c4):
        super(my_gooletnet_block, self).__init__()
        self.incption_1=nn.Sequential(
            nn.Conv2d(inchannel, c1, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.inception_2=nn.Sequential(
            nn.Conv2d(inchannel, c2[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=1),
            nn.ReLU()
        )
        self.inception_3=nn.Sequential(
            nn.Conv2d(inchannel, c3[0], kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], 1),
            nn.ReLU()
        )
        self.inception_4=nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(inchannel, c4, kernel_size=1),
            nn.ReLU()
        )
    def forward(self,input):
        p1=self.incption_1(input)
        p2=self.inception_2(input)
        p3=self.inception_3(input)
        p4=self.inception_4(input)
        out=torch.cat((p1, p2, p3, p4), dim=1)#通道数整合
        return out
class googlenet(nn.Module):
    def __init__(self):
        super(googlenet, self).__init__()
        self.moudle_1=nn.Sequential(
            nn.Conv2d(1,64,7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.moudle_2=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64,192,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.moudle_3=nn.Sequential(
            my_gooletnet_block(192,64,(96,128),(16,32),32),
            my_gooletnet_block(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.moudle_4=nn.Sequential(
            my_gooletnet_block(480,192,(96,208),(16,48),64),
            my_gooletnet_block(512,160,(112,224),(24,64),64),
            my_gooletnet_block(512,128,(128,256),(24,64),64),
            my_gooletnet_block(512,112,(144,288),(32,64),64),
            my_gooletnet_block(528,256,(160,320),(32,128),128),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.moudle_5=nn.Sequential(
            my_gooletnet_block(832,256,(160,320),(32,128),128),
            my_gooletnet_block(832, 384, (192,384), (48,128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.net=nn.Sequential(
            self.moudle_1,
            self.moudle_2,
            self.moudle_3,
            self.moudle_4,
            self.moudle_5,
            nn.Linear(1024,10)
        )
    def forward(self,input):
        out=self.net(input)
        return out

def train(train_l,epoch,optmizer_1):
    for ep in range(1,epoch+1):
        n, train_accu, train_total_loss,train_accu_rate=0,0,0,0
        for input,y_true in train_l:
            input=input.cuda()
            y_true=y_true.cuda()
            y_hat=tran_net(input)
            l1=l(y_hat,y_true)
            train_total_loss+=l(y_hat,y_true)
            optmizer_1.zero_grad()
            l1.backward()
            # l2=list(tran_net.parameters())
             #     print(para.grad)
            optmizer_1.step()
            train_accu+=(y_hat.argmax(dim=1)==y_true).sum().cpu().item()
            n+=y_hat.shape[0]
        train_accu_rate=train_accu/n
        print('epoch:',ep,'total_loss: ',train_total_loss,' train_accurate: ',train_accu_rate)

def test(test_l,epoch):
    for ep in range(1,epoch+1):
        test_total_loss,test_accu,test_accu_rate,n_t=0,0,0,0
        for test_data_in , y_test_true in test_l:
            test_data_in=test_data_in.cuda()
            y_test_true=y_test_true.cuda()
            y_test_hat=tran_net(test_data_in)
            l_test=l(y_test_hat,y_test_true)
            test_total_loss+=l_test
            n_t+=y_test_hat.shape[0]
            test_accu+=(y_test_hat.argmax(dim=1)==y_test_true).float().sum().cpu().item()
        test_accu_rate=test_accu/n_t
        print('epoch: ',ep,'test_total_loss: ',test_total_loss,'test_accu_rate: ',test_accu_rate)
l=nn.CrossEntropyLoss()
tran_net=googlenet()
tran_net.cuda()
# optimizer=torch.optim.SGD(tran_net.parameters(), lr=0.1)
# trans=[transforms.Resize(96),transforms.ToTensor()]
# train_data=torchvision.datasets.FashionMNIST('../data333', train=True,transform=transforms.Compose(trans),download=True)
# trian_l=DataLoader(dataset=train_data,batch_size=300,shuffle=True)
# test_data=torchvision.datasets.FashionMNIST('../data444',train=False,transform=transforms.Compose(trans),download=True)
# test_l=DataLoader(dataset=test_data,batch_size=1000,shuffle=True)
# train(trian_l,10,optimizer)
# test(test_l,5)
print(type(tran_net))
k=nn.Conv2d(1,1,1)
print(type(k))
