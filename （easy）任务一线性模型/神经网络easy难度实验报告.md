### 神经网络简明教程-实践空间站 **实验报告**

## 一、线性回归模型（easy难度）

### 1.**任务描述**

给定含有1000条记录的数据集`mlm.csv`，其中每条记录均包含两个自变量`x`,`y`和一个因变量`z`，它们之间存在较为明显的线性关系。请对数据进行三维可视化分析，并训练处良好的线性回归模型。

### 2.使用的库

- csv
- numpy
- matplotlib.pyplot
- Axes3D

### 3.部分代码及其注释

#读取数据

```python
def ReadData():
    f_csv=csv.reader(open('mlm.csv'))
    headers=next(f_csv)
    #循环获取每一行的内容
    X=[]
    Y=[]
    Z=[]
    #将数据读入数组
    for row in f_csv:
        x=(float)(row[0])
        X.append(x)
        y=(float)(row[1])
        Y.append(y)
        z=(float)(row[2])
        Z.append(z)
    return X,Y,Z
```

#初始化数据

```python
def __init__(self,eta):
	self.eta=eta
    self.k1=0
    self.k2=0
    self.b=0
```

#训练、更新数据

```python
def __update(self,dk1,dk2,db):
    self.k1=self.k1-self.eta*dk1
    self.k2=self.k2-self.eta*dk2
    self.b=self.b-self.eta*db
       
def train(self):
    tot=len(X)
    for i in range(tot):
        x=X[i]
        y=Y[i]
        z=Z[i]
        #print("x=%f,y=%f,z=%f"%(x,y,z))
        for_z = self.__forward(x, y)
        #print("for_z=%f,z=%f"%(for_z,z))
        dk1,dk2,db=self.__backward(x,y,z,for_z)
        self.__update(dk1,dk2,db)
        #print("k1=%f,k2=%f,b=%f"%(self.k1,self.k2,self.b))
```

#三维可视化

```python
def show_3d_predicted_surface(k1,k2,b,X,Y,Z):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.random.randint(0,100, size=20)
    y = np.random.randint(0,100, size=20)
    X1, Y1 = np.meshgrid(x, y)
    Z1 = k1*X1+k2*Y1+b;
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X1, Y1, Z1, cmap='rainbow')
    ax.scatter(X,Y,Z)
    plt.show()
```

#生成训练数据三维散点图

```python
def show_3d_scatter_picture(X,Y,Z):
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.scatter(X,Y,Z)
    plt.show()
```

#主函数

```python
if __name__=='__main__':
    X,Y,Z=ReadData()
    eta=0.0001
    net=NeuralNet_0_1(eta)
    net.train()
    #result
    print("k1=%f,k2=%f,b=%f"%(net.k1,net.k2,net.b))
    print("z=（%f）*x+（%f）*y+（%f）"%(net.k1,net.k2,net.b))
    show_3d_scatter_picture(X,Y,Z)
    show_3d_predicted_surface(net.k1,net.k2,net.b,X,Y,Z)
```

#最终结果及可视化图像

![数据三维散点图](C:\Users\dell\Desktop\learning\Microsoft\Microsoft\数据三维散点图.png)![三维可视化](C:\Users\dell\Desktop\learning\Microsoft\Microsoft\三维可视化.png)![结果](C:\Users\dell\Desktop\learning\Microsoft\Microsoft\结果.png)

（PS：第一个是训练数据的散点图，是程序运行出现的第一张图，第二张图才是三维可视化的图）

最终结果：z=4.088874*x-3.742747*y+0.074204

### 4.完整代码

```python

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ReadData():
    f_csv=csv.reader(open('mlm.csv'))
    headers=next(f_csv)
    #循环获取每一行的内容
    X=[]
    Y=[]
    Z=[]
    #将数据读入数组
    for row in f_csv:
        x=(float)(row[0])
        X.append(x)
        y=(float)(row[1])
        Y.append(y)
        z=(float)(row[2])
        Z.append(z)
    return X,Y,Z


class NeuralNet_0_1(object):
    def __init__(self,eta):
        self.eta=eta
        self.k1=0
        self.k2=0
        self.b=0

    def __forward(self,x,y):
        for_z=x*self.k1+y*self.k2+self.b
        return for_z

    def __backward(self,x,y,z,for_z):
        dz=for_z-z#预测的值减去实际的值
        db=dz
        dk1=x*dz
        dk2=y*dz
        #print("dk1=%f,dk2=%f,db=%f"%(dk1,dk2,db))
        return dk1,dk2,db

    def __update(self,dk1,dk2,db):
        self.k1=self.k1-self.eta*dk1
        self.k2=self.k2-self.eta*dk2
        self.b=self.b-self.eta*db

    def train(self):
        tot=len(X)
        for i in range(tot):
            x=X[i]
            y=Y[i]
            z=Z[i]
            #print("x=%f,y=%f,z=%f"%(x,y,z))
            for_z = self.__forward(x, y)
            #print("for_z=%f,z=%f"%(for_z,z))
            dk1,dk2,db=self.__backward(x,y,z,for_z)
            self.__update(dk1,dk2,db)
            #print("k1=%f,k2=%f,b=%f"%(self.k1,self.k2,self.b))

    def inference(self,x,y):
        return self.__forward(x,y)


def show_3d_predicted_surface(k1,k2,b,X,Y,Z):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.random.randint(0,100, size=20)
    y = np.random.randint(0,100, size=20)
    X1, Y1 = np.meshgrid(x, y)
    Z1 = k1*X1+k2*Y1+b;
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X1, Y1, Z1, cmap='rainbow')
    ax.scatter(X,Y,Z)
    plt.show()

def show_3d_scatter_picture(X,Y,Z):
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.scatter(X,Y,Z)
    plt.show()

if __name__=='__main__':
    X,Y,Z=ReadData()
    eta=0.0001
    net=NeuralNet_0_1(eta)
    net.train()
    #result
    print("k1=%f,k2=%f,b=%f"%(net.k1,net.k2,net.b))
    print("z=（%f）*x+（%f）*y+（%f）"%(net.k1,net.k2,net.b))
    show_3d_scatter_picture(X,Y,Z)
    show_3d_predicted_surface(net.k1,net.k2,net.b,X,Y,Z)
```

