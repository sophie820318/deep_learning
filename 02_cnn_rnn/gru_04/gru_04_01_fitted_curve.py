import numpy as np
#import cupy as np
#使用GPU的场合
import matplotlib.pyplot as plt
#--各项设置参数-
n_time=10
#时间序列数据的数量
n_in=1
#输入层的神经元数量
n_mid=20
#中间层的神经元数量
n_out=1
#输出层的神经元数量
eta=0.01
#学习系数
epochs=101
batch_size =8
interval = 10
#显示处理进度的间隔.sigmoid函数的特点，
# 当输入值较大时，输出值接近1；当输入值较小时，输出值接近0；当输入值为0时，输出值为0.5。
def sigmoid(x):
    return 1/(1+np.exp(-x))

#--创建训练数据--
#从-2π到2π
sin_x=np.linspace(-2*np.pi,2*np.pi)
#使用随机数为sin函数加上噪声
sin_y = np.sin(sin_x) + 0.1 * np.random.randn(len(sin_x))
#n_sample = 40
n_sample = len(sin_x) - n_time
#输入数据，初始化
input_data = np.zeros((n_sample,n_time,n_in))#(4,10,1)
correct_data = np.zeros((n_sample,n_out))
#赋值
for i in range(0,n_sample):
    #reshape(-1, 1)表示把一维切片变成“列向量”，行数自动推断(这里是n_time=10)，所以形状变为(10,1)
    input_data[i] = sin_y[i:i+n_time].reshape(-1,1)
    #正确答案位于输入数据的后一位
    correct_data[i] = sin_y[i+n_time:i+n_time+1]

class GRULayer:
    def __init__(self,n_upper,n):
        #参数的初始值
        self.w = np.random.randn(3, n_upper, n) / np.sqrt(n_upper)
        self.v = np.random.randn(3,n,n) / np.sqrt(n)
    def forward(self,x,y_prev):
        #更新门
        a0= sigmoid(np.dot(x, self.w[0]) +np.dot(y_prev, self.v[0]))
        #复位门
        a1 = sigmoid(np.dot(x, self.w[1]) + np.dot(y_prev, self.v[1]))
        #新的记忆
        a2=np.tanh(np.dot(x, self.w[2]) +np.dot(a1*y_prev, self.v[2]))
        self.gates = np.stack((a0, a1,a2))
        self.y=(1-a0)*y_prev+a0*a2
    #输出数据
    def backward(self,x,y,y_prev, gates,grad_y):
        a0,a1,a2= gates
        #新的记忆
        delta_a2=grad_y *a0*(1-a2**2)
        self.grad_w[2] += np.dot(x.T, delta_a2)
        self.grad_v[2] += np.dot((a1*y_prev).T,delta_a2)
        #更新门
        delta_a0= grad_y *(a2-y_prev) * a0 *(1-a0)
        self.grad_w[0] += np.dot(x.T, delta_a0)
        self.grad_v[0] += np.dot(y_prev.T, delta_a0)
        #复位门
        s=np.dot(delta_a2, self.v[2].T)
        delta_a1=s * y_prev * a1 * (1 - a1)
        self.grad_w[1] +=np.dot(x.T, delta_a1)
        self.grad_v[1] +=np.dot(y_prev.T, delta_a1)
        #X的梯度
        self.grad_x = np.dot(delta_a0, self.w[0].T)
        + np.dot(delta_a1, self.w[1].T)
        + np.dot(delta_a2,self.w[2].T)
        #y_prev的梯度
        self.grad_y_prev = np.dot(delta_a0, self.v[0].T)
        +np.dot(delta_a1, self.v[1].T)
        +a1*s + grad_y*(1-a0)

    def reset_sum_grad(self):
        self.grad_w=np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.v -= eta * self.grad_v
#--全连接输出层-
class OutputLayer:
    def __init__(self, n_upper, n):
        #Xavier的初始值
        self.w = np.random.randn(n_upper,n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x =x
        #对上一层输出x，做的是线性全连接变换：u=xW+b
        u = np.dot(x, self.w) + self.b
        #恒等函数
        self.y =u
    def backward(self,t):
        delta = self.y-t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta,axis=0)
        self.grad_x = np.dot(delta, self.w.T)
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
#--各网络层的初始化
gru_layer = GRULayer(n_in, n_mid)
output_layer = OutputLayer(n_mid,n_out)
#一-训练
def train(x_mb,t_mb):
    #正向传播GRU层
    y_rnn = np.zeros((len(x_mb),n_time+1,n_mid))
    gates_rnn = np.zeros((3, len(x_mb), n_time,n_mid))
    y_prev = y_rnn[:,0,:]
    for i in range(n_time):
        x = x_mb[:, i,:]
        gru_layer.forward(x, y_prev)
        y = gru_layer.y
        y_rnn[:,i+1,:]=y
        y_prev =y
        gates = gru_layer.gates
        gates_rnn[:,:,i,:]=gates
    #正向传播输出层
    output_layer.forward(y)
    #反向传播输出层
    output_layer.backward(t_mb)
    grad_y = output_layer.grad_x
    #反向传播GRU层
    gru_layer.reset_sum_grad()
    for i in reversed(range(n_time)):
        x= x_mb[:,i,:]
        y= y_rnn[:,i+1,:]
        y_prev =y_rnn[:,i,:]
        gates = gates_rnn[:,:, i,:]
        gru_layer.backward(x, y, y_prev, gates, grad_y)
        grad_y = gru_layer.grad_y_prev
    #参数的更新
    gru_layer.update(eta)
    output_layer.update(eta)

def predict(x_mb):
    #正向传播GRU层
    y_prev =np.zeros((len(x_mb),n_mid))
    for i in range(n_time):
        x=x_mb[:,i,:]
        gru_layer.forward(x,y_prev)
        y= gru_layer.y
        y_prev =y
    #正向传播输出层
    output_layer.forward(y)
    return output_layer.y
    #--计算误差-一
def get_error(x, t):
    y= predict(x)
    return 1.0/2.0*np.sum(np.square(y - t)) #误差平方和
error_record=[]
n_batch=len(input_data) // batch_size
#每轮epoch
for i in range(epochs):
    #-学习一一
    index_random = np.arange(len(input_data))
    np.random.shuffle(index_random)
    #打乱索引的顺
    for j in range(n_batch):
        #提取小批次
        mb_index = index_random[j*batch_size :(j+1)*batch_size]
        x_mb = input_data[mb_index,:]
        t_mb = correct_data[mb_index,:]
        train(x_mb,t_mb)
    #--计算误差一
    error = get_error(input_data, correct_data)
    error_record.append(error)
    #--显示处理进度一一
    if i%interval ==0:
        print("Epoch:"+str(i+1)+"/"+str(epochs), "Error:"+str(error))
        #开始的输入
        predicted = input_data[0].reshape(-1).tolist()
        for i in range(n_sample):
            x=np.array(predicted[-n_time:]).reshape(1, n_time, 1)
            y= predict(x)
            predicted.append(float(y[0,0]))
        plt.plot(range(len(sin_y)),sin_y.tolist(), label="Correct")
        plt.plot(range(len(predicted)),predicted, label="Predicted")
        plt.legend()
        plt.show()
plt.plot(range(1, len(error_record)+1), error_record)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
