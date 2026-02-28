import numpy as np
#使用GPU的场合
import matplotlib.pyplot as plt
#--各项设置参数
#时间序列的数量
n_time=10
#输入层的神经元数量
n_in=1
#中间层的神经元数量
n_mid=20
#输出层的神经元数量
n_out=1
#学习系数
eta=0.001
epochs=51
batch_size =8
#显示处理进度的间隔
interval=5
#--生成训练数据集
#默认包含50个在-2π到2π之间等间隔的点
sin_x= np.linspace(-2*np.pi,2*np.pi)
#使用随机数向sin函数中添加噪声
sin_y = np.sin(sin_x) + 0.1*np.random.randn(len(sin_x))
#样本数量：50-10=40
n_sample = len(sin_x)-n_time#50-10=40
# n_sample：样本数量
# n_time：每个样本的时间步长度（窗口长度）
# n_in：每个时间步的输入特征维度
#(40,10,1)=400个数据
input_data = np.zeros((n_sample,n_time,n_in)) #输入数据，NumPy的ndarray三维数组
#正确答案(40,1)
correct_data = np.zeros((n_sample,n_out))
#每10步一个样本集合初始化样本
for i in range (0,n_sample):
    #1：第二维固定为 1 列
    #-1：第一维由 NumPy 自动计算，使元素总数不变。
    # 因此它会把一维长度为n_time的数组变成形状 (n_time, 1)
    #这是结果
    input_data[i] = sin_y[i:i+n_time].reshape(-1,1) 
    #input_data的下一个结果
    correct_data[i]= sin_y[i+n_time:i+n_time+1]
#正确答案位于输入后的一位

#rnn
class SimpleRNNLayer:
    def __init__(self,n_upper,n):
        #参数的初始值
        #Xavier的初始值
        self.w =np.random.randn(n_upper,n)/ np.sqrt(n_upper)
        #Xavier的初始值
        #self.v 是循环权重矩阵（隐藏层到隐藏层的权重）,用于把上一个时刻的隐藏状态 y_prev 
        #参与当前时刻的计算：u = x·w + y_prev·v + b。没有 self.v 就不是RNN,而是普通前馈层.
        self.v = np.random.randn(n,n) / np.sqrt(n)
        self.b = np.zeros(n)

    def forward(self,x,y_prev):
        #y_prev:前一时刻的输出
        u = np.dot(x, self.w) + np.dot(y_prev, self.v) + self.b
        #激活函数是双曲正切函数tanh，输出范围在-1到1之间
        self.y= np.tanh(u)
#输出
    def backward(self,x, y,y_prev, grad_y):
        delta= grad_y*(1-y**2)
        #各个梯度
        #在 BPTT：Backpropagation Through Time中，每个时间步都会对同一组参数产生梯度，
        #需要把各时间步（以及小批次样本）的梯度累加后再更新，所以用 += 累积总梯度
        self.grad_w += np.dot(x.T, delta)
        self.grad_v += np.dot(y_prev.T,delta)
        self.grad_b += np.sum(delta,axis=0)
        self.grad_x = np.dot(delta, self.w.T)
        self.grad_y_prev = np.dot(delta, self.v.T)
    def reset_sum_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)

    def update(self,eta):
        self.w -= eta * self.grad_w
        self.v -= eta * self.grad_v
        self.b -= eta * self.grad_b
class OutputLayer:
    def __init__(self, n_upper, n):
        #Xavier的初始值
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x =x
        u = np.dot(x, self.w) + self.b
        self.y=u
    def backward(self,t):
        delta =self.y-t
        self.grad_w = np.dot(self.x.T,delta)
        self.grad_b = np.sum(delta,axis=0)
        self.grad_x= np.dot(delta,self.w.T)
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
    
#各个网络层的初始化一一
rnn_layer = SimpleRNNLayer(n_in,n_mid)
output_layer = OutputLayer(n_mid,n_out)
#—-训练———
def train(x_mb,t_mb):
    #正向传播 RNN层
    y_rnn = np.zeros((len(x_mb),n_time+1,n_mid))
    y_prev =y_rnn[:,0,:]
    for i in range(n_time):
        #第 i 个时间步的输入。x_mb 形状是 (batch, n_time, n_in)，
        #所以 x_mb[:, i, :] 得到形状 (batch, n_in)，表示所有样本在第 i 个时间步的特征。
        x=x_mb[:,i,:]
        rnn_layer.forward(x, y_prev)
        y = rnn_layer.y
        y_rnn[:,i+1,:] =y
        y_prev =y
    # 是“序列→单值”的回归，只用最后一个时间步的隐藏状态做输出，
    # 所以 output_layer.forward(y) 放在时间循环结束后（只算一次）。
    output_layer.forward(y)
    #反向传播输出层
    output_layer.backward(t_mb)
    #这个 grad_x 就是对输出层输入(也就是最后时刻隐藏状态y)的梯度
    grad_y = output_layer.grad_x
    #反向传播 RNN层
    rnn_layer.reset_sum_grad()
    for i in reversed(range(n_time)):
        x = x_mb[:, i, :]
        y = y_rnn[:,i+1,:]
        y_prev = y_rnn[:, i, :]
        rnn_layer.backward(x, y, y_prev, grad_y)
        grad_y = rnn_layer.grad_y_prev
    #参数的更新
    rnn_layer.update(eta)
    output_layer.update(eta)
#--预测-一
def predict(x_mb):
    #正向传播RNN层
    y_prev= np.zeros((len(x_mb),n_mid))
    for i in range(n_time):
        x = x_mb[:, i, :]
        rnn_layer.forward(x, y_prev)
        y = rnn_layer.y
        y_prev = y
    #正向传播输出层
    output_layer.forward(y)
    return output_layer.y
#--计算误差--
def get_error(x,t):
    y = predict(x)
    return 1.0/2.0*np.sum(np.square(y - t)) #误差平方和


error_record = []
n_batch = len(input_data) // batch_size   #每个epoch的批次数量，除法取整。40/8 = 5
    #每个epoch的批次数量
for i in range(epochs):#51
    #--学习一一
    index_random = np.arange(len(input_data))  
    np.random.shuffle(index_random)
    #将索引打乱
    for j in range(n_batch):#
        #取出小批次数据进行训练8-
        mb_index= index_random[j*batch_size:(j+1)*batch_size]#batch_size=8
        x_mb = input_data[mb_index,:]
        t_mb = correct_data[mb_index,:]
        train(x_mb,t_mb)
    error = get_error(input_data, correct_data)
    error_record.append(error)
    #--显示处理进度-一
    if i%interval == 0:#interval=5,每隔5次显示一次
        print("Epoch:"+str(i+1)+"/"+str(epochs),"Error:"+str(error))
        predicted = input_data[0].reshape(-1).tolist() #起始的输入  
        for i in range(n_sample):#40,n_time=10
            x=np.array(predicted[-n_time:]).reshape(1,n_time,1)
            y=predict(x)
            #将输出添加到predicted中
            predicted.append(float(y[0,0]))
        plt.plot(range(len(sin_y)),sin_y.tolist(),label="Correct")
        plt.plot(range(len(predicted)),predicted,label="Predicted")
        plt.legend()
        plt.show()
plt.plot(range(1, len(error_record)+1), error_record)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
