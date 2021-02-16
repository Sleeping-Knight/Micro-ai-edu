# Micro-ai-edu

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
## 二、非线性多分类实验（medium难度）

### 1.**任务描述**

鸢尾花数据集iris.csv含有150条记录，每条记录包含萼片长度sepal length、萼片宽度sepal width、 花瓣长度petal length和花瓣宽度petal width四个数值型特征，以及它的所属类别class（可能为Iris-setosa,Iris-versicolor,Iris-virginica三者之一）。请利用该数据集训练出一个良好的非线性分类器。
### 2.使用的库

- pandas
- numpy
- sklearn

### 3.部分代码及其注释

#读取数据

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'iris.csv'
Data = pd.read_csv(path)

cols = Data.shape[1]  # Data.shape[0]表示的是数据的行数，Data.shape[1]表示的是数据的列数
X = Data.iloc[:, :cols - 1]  # 表示X读取了除最后一列的所有数据
X = np.array(X)
iris_class = Data.iloc[:, cols - 1:cols]  # iris_class表示鸢尾花的分类
iris_class = np.array(iris_class)


#处理数据，将鸢尾花的名字映射为0,1,2,三个数字

```python
def iris_name_to_int(string_arr):
    int_arr = []
    for i in range(len(string_arr)):
        if (string_arr[i] == 'Iris-setosa'):
            int_arr.append(0)
        if (string_arr[i] == 'Iris-versicolor'):
            int_arr.append(1)
        if (string_arr[i] == 'Iris-virginica'):
            int_arr.append(2)
    int_arr = np.array(int_arr)
    return int_arr


y = iris_name_to_int(iris_class)
```

#样本特征数据标准化

```python
ef NormalizeX():
    for i in range(X.shape[1]):
        M = np.max(X[:, i:i + 1])
        m = np.min(X[:, i:i + 1])
        for j in range(X.shape[0]):
            X[j][i] = (X[j][i] - m) / (M - m)


NormalizeX()
```

#划分训练集和数据集并设定学习率和激活函数

```python
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 初始化权重值
W1 = np.random.normal(0, 1, (4, 8))
W2 = np.random.normal(0, 1, (8, 3))

# 设定学习率
eta = 0.01


# Sigmoid函数
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

```

#前向传播，反向传播以及更新数据

```python
# 前向传播
def forward(X):
    Z1 = np.dot(X, W1)
    A1 = Sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = Sigmoid(Z2)
    return Z1, A1, Z2, A2


# 反向传播
def backward(A1, A2):
    dZ2 = A2 - y_true
    dW2 = np.dot(A1.T, dZ2)
    tmp = np.dot(dZ2, W2.T)
    dW1 = np.dot(X_train.T, tmp * A1 * (1 - A1))
    return dW1, dW2


# 更新
def update(dW1, dW2):
    global W1, W2
    W1 -= dW1 * eta
    W2 -= dW2 * eta


# 神经网络完成一整个流程
def Neural_Net_Main_function():
    Z1, A1, Z2, A2 = forward(X_train)
    dW1, dW2 = backward(A1, A2)
    update(dW1, dW2)
```

#统计训练准确率

```python
# 统计训练准确率
def learn_score(y_test_true, y_predict):
    return sum(y_test_true == y_predict) / len(y_test_true)


OneHot = np.identity(3)
y_true = OneHot[y_train]
```

#设定训练次数，开始训练

```python
# 设定训练次数
train_times = 10000

# 开始训练
for i in range(train_times):
    Neural_Net_Main_function()
```

#对测试集进行预测并打印预测准确率

```python
# 对测试集预测
Z1_test = np.dot(X_test, W1)
A1_test = Sigmoid(Z1_test)
Z2_test = np.dot(A1_test, W2)
A2_test = Sigmoid(Z2_test)

y_test_predict = []
for i in range(A2_test.shape[0]):
    iris_index = np.argmax(A2_test[i])  # 概率函数，返回最大值的index
    y_test_predict.append(iris_index)
y_test_predict = np.array(y_test_predict)

#打印预测准确率
print(learn_score(y_test, y_test_predict))

```

#结果示意图

![结果1](C:\Users\dell\Desktop\learning\Iris_non_linear_classifier\Iris_non_linear_classifier\Iris_non_linear_classifier\结果1.png)

![结果2](C:\Users\dell\Desktop\learning\Iris_non_linear_classifier\Iris_non_linear_classifier\Iris_non_linear_classifier\结果2.png)

![结果3](C:\Users\dell\Desktop\learning\Iris_non_linear_classifier\Iris_non_linear_classifier\Iris_non_linear_classifier\结果3.png)

总体预测准确率稳定在90%以上，在95%左右。

### 4.完整代码

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'iris.csv'
Data = pd.read_csv(path)

cols = Data.shape[1]  # Data.shape[0]表示的是数据的行数，Data.shape[1]表示的是数据的列数
X = Data.iloc[:, :cols - 1]  # 表示X读取了除最后一列的所有数据
X = np.array(X)
iris_class = Data.iloc[:, cols - 1:cols]  # iris_class表示鸢尾花的分类
iris_class = np.array(iris_class)


# 为了处理问题方便，将鸢尾花的名字映射为0,1,2三个数字
def iris_name_to_int(string_arr):
    int_arr = []
    for i in range(len(string_arr)):
        if (string_arr[i] == 'Iris-setosa'):
            int_arr.append(0)
        if (string_arr[i] == 'Iris-versicolor'):
            int_arr.append(1)
        if (string_arr[i] == 'Iris-virginica'):
            int_arr.append(2)
    int_arr = np.array(int_arr)
    return int_arr


y = iris_name_to_int(iris_class)


# 样本特征数据标准化
def NormalizeX():
    for i in range(X.shape[1]):
        M = np.max(X[:, i:i + 1])
        m = np.min(X[:, i:i + 1])
        for j in range(X.shape[0]):
            X[j][i] = (X[j][i] - m) / (M - m)


NormalizeX()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 初始化权重值
W1 = np.random.normal(0, 1, (4, 8))
W2 = np.random.normal(0, 1, (8, 3))

# 设定学习率
eta = 0.01


# Sigmoid函数
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 前向传播
def forward(X):
    Z1 = np.dot(X, W1)
    A1 = Sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = Sigmoid(Z2)
    return Z1, A1, Z2, A2


# 反向传播
def backward(A1, A2):
    dZ2 = A2 - y_true
    dW2 = np.dot(A1.T, dZ2)
    tmp = np.dot(dZ2, W2.T)
    dW1 = np.dot(X_train.T, tmp * A1 * (1 - A1))
    return dW1, dW2


# 更新
def update(dW1, dW2):
    global W1, W2
    W1 -= dW1 * eta
    W2 -= dW2 * eta


# 神经网络完成一整个流程
def Neural_Net_Main_function():
    Z1, A1, Z2, A2 = forward(X_train)
    dW1, dW2 = backward(A1, A2)
    update(dW1, dW2)


# 统计训练准确率
def learn_score(y_test_true, y_predict):
    return sum(y_test_true == y_predict) / len(y_test_true)


OneHot = np.identity(3)
y_true = OneHot[y_train]

# 设定训练次数
train_times = 10000

# 开始训练
for i in range(train_times):
    Neural_Net_Main_function()

# 对测试集预测
Z1_test = np.dot(X_test, W1)
A1_test = Sigmoid(Z1_test)
Z2_test = np.dot(A1_test, W2)
A2_test = Sigmoid(Z2_test)

y_test_predict = []
for i in range(A2_test.shape[0]):
    iris_index = np.argmax(A2_test[i])  # 概率函数，返回最大值的index
    y_test_predict.append(iris_index)
y_test_predict = np.array(y_test_predict)

#打印预测准确率
print(learn_score(y_test, y_test_predict))
```


