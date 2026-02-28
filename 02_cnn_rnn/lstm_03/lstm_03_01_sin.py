import numpy as np
#使用GPU的场合
#import cupy as np
import matplotlib.pyplot as plt
#--各项设置参数-
#时间序列数据的数量
n_time = 10
#输入层的神经元数量
n_in=1
#中间层的神经元数量
n_mid=20
#输出层的神经元数量
n_out=1
eta=0.01
#学习系数
epochs= 101
batch_size = 8
interval=10
#显示处理进度的间隔
def sigmoid(x):
    return 1/(1+np.exp(-x))
#--训练数据的生成一
#默认包含50个在-2π到2π之间等间隔的点
sin_x = np.linspace(-2*np.pi, 2*np.pi)
#使用随机数向sin函数中添加噪声
#从-2π到2π
sin_y = np.sin(sin_x) + 0.1*np.random.randn(len(sin_x))
n_sample = len(sin_x)-n_time#40
input_data = np.zeros((n_sample, n_time, n_in)) #(40,10,1)
#样本数量
#输入数据
correct_data = np.zeros((n_sample, n_out))#(40,1)
#正确答案
for i in range(0,n_sample):
    #reshape(-1, 1) 表示把一维切片变成“列向量”，
    #行数自动推断(这里是n_time=10)所以形状变为 (10, 1)
    input_data[i]= sin_y[i:i+n_time].reshape(-1,1)
    #正确答案位于输入数据的后一位
    correct_data[i] = sin_y[i+n_time:i+n_time+1]
#--LSTM网络层-
class LSTMLayer:
    def __init__(self,n_upper,n):
        #各项参数的初始值
        self.w = np.random.randn(4,n_upper, n) / np.sqrt(n_upper)
        self.v = np.random.randn(4,n,n) / np.sqrt(n)
        self.b = np.zeros((4, n))
    #y-prev,c_prev：前一时刻的输出数据与记忆单元
    def forward(self,x,y_prev,c_prev):
        u = np.matmul(x, self.w) + np.matmul(y_prev,self.v) + self.b.reshape(4, 1, -1)
        #忘记门
        a0 = sigmoid(u[0])
        #输入门
        a1 = sigmoid(u[1])
        #候选记忆
        a2=np.tanh(u[2])
        #输出门
        a3 =sigmoid(u[3])
        # (4, batch, n_mid) 的 self.gates
        self.gates = np.stack((a0,a1,a2, a3))
        #忘记门*记忆单元+输入门*新的记忆
        #对记忆单元c,c_prev:上一时刻记忆c_prev 
        self.c = a0*c_prev + a1*a2
        #输出数据
        self.y = a3 * np.tanh(self.c)
    def backward(self,x,y,c,y_prev, c_prev, gates,grad_y,grad_c):
        a0,a1,a2,a3 = gates #解包赋值
        #对记忆单元 c 做双曲正切，将其压到(−1,1)
        #(−1,1) 范围内。这样输出更稳定、梯度更平滑，后面用于计算输出门相关的值。
        tanh_c = np.tanh(c)
        r=grad_c+(grad_y*a3) *(1-tanh_c**2)
        #各项delta
        delta_a0=r*c_prev*a0*(1-a0)
        delta_a1=r*a2*a1*(1-a1)
        delta_a2=r*a1*(1-a2**2)
        delta_a3=grad_y*tanh_c*a3*(1-a3)
        deltas =np.stack((delta_a0, delta_a1, delta_a2, delta_a3))
        #各项参数的梯度
        self.grad_w += np.matmul(x.T,deltas)
        self.grad_v += np.matmul(y_prev.T, deltas)
        self.grad_b += np.sum(deltas,axis=1)
        #×的梯度
        #self.w 的最后两维从 (n_upper, n) 变成 (n, n_upper)，这样deltas的形状 (4, batch, n) 
        # 才能与之矩阵乘法，得到 (4, batch, n_upper) 的 grad_x
        # transpose(0,2,1) 表示对数组做维度置换：第0维保持不变,把第1维和第2维交换。
        # 也就是将形状 (4, n_upper, n) 变为 (4, n, n_upper)，用于匹配矩阵乘法的维度。
        grad_x = np.matmul(deltas, self.w.transpose(0,2,1))
        self.grad_x = np.sum(grad_x, axis=0)
        #y_prev的梯度
        grad_y_prev = np.matmul(deltas, self.v.transpose(0,2,1))
        self.grad_y_prev = np.sum(grad_y_prev, axis=0)
        #c_prev的梯度
        self.grad_c_prev = r * a0
    def reset_sum_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.v -= eta * self.grad_v
        self.b -= eta * self.grad_b
class OutputLayer:
    def __init__(self,n_upper,n):
        self.w = np.random.randn(n_upper,n)/np.sqrt(n_upper)
        #Xavier的初始
        self.b = np.zeros(n)
    def forward(self,x):
        self.x = x
        u = np.dot(x,self.w) + self.b
        self.y =u
    #恒等函数
    def backward(self, t):
        delta=self.y-t
        self.grad_w =np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
#一-各网络层的初始化一一
lstm_layer =LSTMLayer(n_in,n_mid)
output_layer = OutputLayer(n_mid,n_out)

#--训练函数一一
def train(x_mb, t_mb):
    #正向传播LSTM网络层,隐藏层的输出y和记忆单元c在每个时间步都要保存下来，
    # 以便后面反向传播时使用
    y_rnn = np.zeros((len(x_mb),n_time+1,n_mid))#隐藏层的输出
    c_rnn = np.zeros((len(x_mb),n_time+1,n_mid))
    gates_rnn = np.zeros((4,len(x_mb),n_time,n_mid))
    y_prev=y_rnn[:,0,:]
    c_prev =c_rnn[:,0,:]
    for i in range(n_time):
        x=x_mb[:,i,:]
        lstm_layer.forward(x, y_prev, c_prev)
        y = lstm_layer.y
        y_rnn[:,i+1,:] =y
        y_prev = y
        c = lstm_layer.c
        c_rnn[:,i+1,:] =c
        c_prev=c
        gates = lstm_layer.gates
        gates_rnn[:,:,i,:] = gates
    output_layer.forward(y)
    #反向传播输出层
    output_layer.backward(t_mb)
    grad_y = output_layer.grad_x
    grad_c = np.zeros_like(lstm_layer.c)
    #反向传播LSTM网络层
    lstm_layer.reset_sum_grad()
    for i in reversed(range(n_time)):
        x = x_mb[:, i, :]
        y = y_rnn[:, i+1, :]
        c = c_rnn[:, i+1, :]
        y_prev = y_rnn[:, i, :]
        c_prev = c_rnn[:, i, :]
        gates = gates_rnn[:, :, i, :]
        lstm_layer.backward(x, y, c, y_prev, c_prev, gates, grad_y, grad_c)
        grad_y = lstm_layer.grad_y_prev
        grad_c = lstm_layer.grad_c_prev
        #参数的更新
    lstm_layer.update(eta)
    output_layer.update(eta)

def predict(x_mb):
    #正向传播LSTM网络层
    y_prev = np.zeros((len(x_mb),n_mid))
    c_prev = np.zeros((len(x_mb),n_mid))
    for i in range(n_time):
        x=x_mb[:,i,:]
        lstm_layer.forward(x,y_prev,c_prev)
        y = lstm_layer.y
        y_prev =y
        c = lstm_layer.c
        c_prev =c
    #正向传播输出层
    output_layer.forward(y)
    return output_layer.y
    #--计算误差
    一
def get_error(x,t):
    y= predict(x)
    return 1.0/2.0*np.sum(np.square(y-t))
error_record =[]
#每轮ey
n_batch = len(input_data) // batch_size
for i in range(epochs):
    #一一学习一一
    index_random = np.arange(len(input_data))
    #对索引进
    np.random.shuffle(index_random)
    for j in range(n_batch):
        #取出小批次数据
        mb_index = index_random[j*batch_size : (j+1)*batch_size]
        x_mb = input_data[mb_index,:]
        t_mb = correct_data[mb_index,:]
        train(x_mb,t_mb)
    #--计算误差-一
    error =get_error(input_data,correct_data)
    error_record.append(error)
    #--显示进度--
    if i%interval == 0:
        print("Epoch:" + str(i+1) + "/" + str(epochs),"Error:" + str(error))
        predicted = input_data[0].reshape(-1).tolist()#最初的输
        for i in range(n_sample):
            x = np.array(predicted[-n_time:]).reshape(1,n_time,1)
            y= predict(x)
            #将输出添加到predicted中
            predicted.append(float(y[0,0]))
        plt.plot(range(len(sin_y)), sin_y.tolist(), label="Correct")
        plt.plot(range(len(predicted)),predicted,label="Predicted")
        plt.legend()
        plt.show()
plt.plot(range(1, len(error_record)+1), error_record)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
