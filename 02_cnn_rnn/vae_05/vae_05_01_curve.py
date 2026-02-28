import numpy as np
#使用GPU的场合
#import cupy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#--设置各项参数

#图像的高度和宽度
img_size =8
#输入层和输出层的神经元数量
n_in_out = img_size * img_size
#中间层的神经元数量
n_mid=16
#学习系数
eta=0.01
epochs=41
# 显示处理进度的间隔
batch_size =32
interval=4
#--训练数据
digits_data = datasets.load_digits()
#形状约为 (1797, 64)，每行是一个 8×8 图像展平后的 64 维特征向量。
# 初始是数值型（通常是 float64），下一行做 /= 15.0 后为归一化浮点数。
x_train = np.array(digits_data.data)
x_train/= 15.0
class BaseLayer:
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
class MiddleLayer(BaseLayer):
    def __init__(self,n_upper,n):
        #He的初始值
        #He 初始化是一种权重初始化方法，适配 ReLU 类激活函数。它将权重按照np.sqrt(2/n_upper)进行缩放，
        # 以保持前向传播时的信号稳定，避免梯度消失或爆炸问题。
        self.w = np.random.randn(n_upper,n) * np.sqrt(2/n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x=x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0,0, self.u) #ReLU
    def backward(self,grad_y):
        delta = grad_y * np.where(self.u <= 0,0,1)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
    #--输出层--
class OutputLayer(BaseLayer):
    def __init__(self,n_upper,n):
        #Xavier的初始值
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x =x
        u = np.dot(x, self.w) + self.b
        #Sigmoid函数
        # Sigmoid 用来把任意实数映射到 (0,1)，常用于输出层表示概率或将输出压到固定范围。
        # 这里用于重构输出，使像素值在 0∼1 之间，并且它是可导的便于反向传播。
        self.y =1/(1+np.exp(-u))
    def backward(self,t):
        delta = (self.y-t) * self.y * (1-self.y)
        self.grad_w= np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
        #--各网络层的初始化--
        #Encoder
middle_layer = MiddleLayer(n_in_out,n_mid)
#Decoder
output_layer = OutputLayer(n_mid,n_in_out)
def forward_propagation(x_mb):
    middle_layer.forward(x_mb)
    output_layer.forward(middle_layer.y)
#一一反向传播
def backpropagation(t_mb):
    output_layer.backward(t_mb)
    middle_layer.backward(output_layer.grad_x)
#--参数的更新--
def update_params():
    middle_layer.update(eta)
    output_layer.update(eta)
#--计算误差
def get_error(y,t):
    return 1.0/2.0*np.sum(np.square(y -t))
#误差平方和
error_record =[]
#这是向下取整的整除运算
n_batch = len(x_train) // batch_size
#每轮epoch的批次
for i in range(epochs):
    #--学习一一
    index_random = np.arange(len(x_train))
    np.random.shuffle(index_random)
    #打乱索引的顺序
    for j in range(n_batch):
        #提取小批次数据
        mb_index = index_random[j*batch_size :(j+1)*batch_size]
        x_mb = x_train[mb_index,:]
        #正向传播与反向传播
        #第一次在每个小批次训练时做前向传播，用于计算输出并进行反向传播更新参数；
        forward_propagation(x_mb)
        backpropagation(x_mb)
        #权重与偏置的更新
        update_params()
    #第二次在每个 epoch 结束后对全训练集前向传播，用来计算并记录整体误差用于绘图。
    forward_propagation(x_train)
    error = get_error(output_layer.y,x_train)
    error_record.append(error)
    #--显示实现过程-
    if i%interval == 0:
        print("Epoch:"+str(i+1)+"/"+str(epochs),"Error:"+str(error))
plt.plot(range(1,len(error_record)+1),error_record)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

