
import numpy as np
import argparse
import torch as t
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


from sklearn.datasets.samples_generator import make_blobs


#构建网络
class SVM(nn.Module):
    def __init__(self):
        super(SVM,self).__init__()
        self.f = nn.Linear(2,1)    #输入特征2 输出特征1 矩阵2x1

    def forward(self,x):
        h = self.f(x)
        return h

def trainSVM(X,Y):
    model = SVM() #自定义的线性svm
    if t.cuda.is_available():  #使用gpu
        model.cuda()

    X = t.FloatTensor(X)  #转换成张量
    Y = t.FloatTensor(Y)

    N= len(Y)

    optimizer = optim.SGD(model.parameters(),lr=0.05)  #lr学习速率 随机梯度下降

    model.train()
    for epoch in range(20):
        perm = t.randperm(N) #随机选取0到N-1
        sum_loss = 0

        for i in range(0,N,1): #1 - betchsize
            x = X[perm[i:i+1]]
            y = Y[perm[i:i+1]]

            if t.cuda.is_available():
                x=x.cuda()
                y=y.cuda()

            optimizer.zero_grad()
            output = model(x)

            loss = t.mean(t.clamp(1-output.t()*y,min=0))  #hinge loss
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().detach().numpy()

        print("Epoch:{:4d}\tloss:{}".format(epoch, sum_loss / N))
    print(model.f.weight.cpu().detach().numpy()[0])

    return model.f.weight.cpu().detach().numpy()[0],model.f.bias.cpu().detach().numpy()[0]

def plot_after_train(X,Y,weight,bias):
    x,y=X,Y
    for i in range(y.size):
        plt.scatter(x[i][0],x[i][1],color='red' if y[i]==1 else 'blue')

    # delta = 0.01
    # x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    # y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    # x, y = np.meshgrid(x, y)
    b = np.arange(-1.5,1)

    print(weight[0])

    a = weight[0]* b + bias

    print(b)
    print("a:{}",format(a))

    plt.plot(b,a)

    plt.show()

def plot_after_train_1(X,Y,weight,bias):
    W = weight
    b = bias
    x, y = X, Y

    for i in range(y.size):
        plt.scatter(x[i][0],x[i][1],color='red' if y[i]==1 else 'blue')

    delta = 0.01
    x = np.arange(X[:, 0].min()-0.5, X[:, 0].max()+0.5, delta)
    y = np.arange(X[:, 1].min()-0.5, X[:, 1].max()+0.5, delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)

    z[np.where(z > 1.)] = 4
    z[np.where((z > 0.) & (z <= 1.))] = 3
    z[np.where((z > -1.) & (z <= 0.))] = 2
    z[np.where(z <= -1.)] = 1

    plt.contourf(x, y, z, alpha=0.6, cmap="Greys")

    plt.tight_layout()
    plt.show()




def visualize(X, Y, weight,bias):
    W = weight
    b = bias

    delta = 0.01
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)

    print("z-before:",format(z))
    z[np.where(z > 1.)] = 4
    z[np.where((z > 0.) & (z <= 1.))] = 3
    z[np.where((z > -1.) & (z <= 0.))] = 2
    z[np.where(z <= -1.)] = 1

    print("z-after:",format(z))

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
    plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10) #画点
    plt.tight_layout()
    plt.show()













def get_data():
    X,Y=make_blobs(n_samples=500,centers=2,random_state=0,cluster_std=0.8)

    Y[Y==0]=-1  #label为0的变成-1 便于训练
    X = (X-X.mean())/X.std()
    return X,Y   # X - 参数坐标  Y - 参数label （-1,1）

def Plot_Data(X,Y):
    x,y=X,Y
    for i in range(y.size):
        plt.scatter(x[i][0],x[i][1],color='red' if y[i]==1 else 'blue')
    plt.show()





if __name__ == '__main__':
    X,Y=get_data()
    Plot_Data(X,Y)
    weight,bias = trainSVM(X,Y)
    visualize(X,Y,weight,bias)

    plot_after_train(X,Y,weight,bias)

    plot_after_train_1(X,Y,weight,bias)

