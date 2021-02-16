### 神经网络简明教程-实践空间站 **实验报告**

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
path = 'iris.csv'
Data = pd.read_csv(path)

cols = Data.shape[1]  # Data.shape[0]表示的是数据的行数，Data.shape[1]表示的是数据的列数
X = Data.iloc[:, :cols - 1]  # 表示X读取了除最后一列的所有数据
X = np.array(X)
iris_class = Data.iloc[:, cols - 1:cols]  # iris_class表示鸢尾花的分类
iris_class = np.array(iris_class)
```

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
    return sum(y_test_true == y_predict) / len(y_true)


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
    return sum(y_test_true == y_predict) / len(y_true)


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

