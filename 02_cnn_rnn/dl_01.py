import numpy as np
#使用GPU的场合
#import cupy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
#--各项设置参数-
img_size =8
n_mid=16
n_out=10
eta=0.001
epochs =51
batch_size = 32
interval =5
#显示处理进度的间
digits_data = datasets.load_digits()
#一一输入数据-
input_data = np.asarray(digits_data.data)
#平均值为0、标准偏差为1
input_data = (input_data - np.average(input_data)) / np.std(input_data)
#--正确答案数据--
correct = np.asarray(digits_data.target)
correct_data = np.zeros((len(correct), n_out))
for i in range(len(correct)):
    correct_data[i, correct[i]] = 1 #独热格式
#一-分割为训练数据和测试数据-
x_train,x_test,t_train, t_test = \
    train_test_split(input_data,correct_data)
#全连接层的父类
class BaseLayer:
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
class MiddleLayer(BaseLayer):
    def __init__(self,n_upper,n):
    #He的初始值_
        self.w = np.random.randn(n_upper, n) * np.sqrt(2/n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x=x
        self.u = np.dot(x, self.w) + self.b
        self.y =np.where(self.u <=0,0,self.u)
    #ReLU
    def backward(self,grad_y):
        delta = grad_y *np.where(self.u <= 0, 0, 1) #relu的微分
        #ReLU的微分
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

class OutputLayer(BaseLayer):
    def __init__(self,n_upper,n):
        #Xavier的初始值
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x = x
        u = np.dot(x,self.w) + self.b
#Softmax函数
        self.y = np.exp(u)/np.sum(np.exp(u),axis=1,keepdims=True)
    def backward(self,t):
        delta = self.y-t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
#一各个网络层的初始化一
layers=[MiddleLayer(img_size*img_size,n_mid),
    MiddleLayer(n_mid, n_mid),
    OutputLayer(n_mid, n_out)]
#一正向传播
def forward_propagation(x):
    for layer in layers:
        layer.forward(x)
        x=layer.y
    return x
#--反向传播
def backpropagation(t):
    grad_y = t
    for layer in reversed(layers):
        layer.backward(grad_y)
        grad_y = layer.grad_x
    return grad_y
#--参数的更新
def update_params():
    for layer in layers:
        layer.update(eta)

def get_error(x,t):
    y=forward_propagation(x)
    return -np.sum(t*np.log(y+1e-7)) / len(y) #交叉熵误差

#一-准确率的测定
def get_accuracy(x,t):
    y=forward_propagation(x)

    count =np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1))
    return count / len(y)
#--误差的记录-
error_record_train =[]
error_record_test = []
n_batch =len(x_train) // batch_size   #每轮epoch的批次大小
for i in range (epochs):
    index_random = np.arange(len(x_train))
    np.random.shuffle(index_random)
    #打乱索引的顺序
    for j in range(n_batch):
        #提取小批次数据
        mb_index =index_random[j*batch_size : (j+1)*batch_size]
        x_mb = x_train[mb_index,:]
        t_mb = t_train[mb_index,:]
        #正向传播与反向传播
        forward_propagation(x_mb)
        backpropagation(t_mb)
        #参数的更新
        update_params()
    #--误差的测量和记录一
    error_train = get_error(x_train,t_train)
    error_record_train.append(error_train)
    error_test = get_error(x_test,t_test)
    error_record_test.append(error_test)
    #--显示处理进度--
    if i%interval == 0:
        print("epoch:"+str(i)+"/"+str(epochs),
        "Error_train"+str(error_train))

        "Error_test: " + str(error_test)
#--用图表显示误差的推移--
plt.plot(range(1, len(error_record_train)+1),
    error_record_train,label="Train")
plt.plot(range(1, len(error_record_test)+1),
    error_record_test,label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
#--准确率的测定--
acc_train = get_accuracy(x_train,t_train)
acc_test = get_accuracy(x_test, t_test)
print("Acc train:"+str(acc_train*100)+"%"
    "Acc test:"+str(acc_test*100)+"%")
