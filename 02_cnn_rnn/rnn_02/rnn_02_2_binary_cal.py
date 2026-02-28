import numpy as np
#import cupy as np #使用GPU的场合
n_time =8
#输入层的神经元数量
n_in=2
n_mid=32
#中间层的神经元数量
n_out=1
#输出层的神经元数量
eta=0.01
#学习系数
#学习次数
n_learn =5001
interval= 500
#显示进度的间隔
#--生成二进制数一-
max_num = 2**n_time#2**8 =256
binaries = np.zeros((max_num, n_time),dtype=int)# (256,8)
#十进制数的上限
for i in range(max_num):
    num10=i#10进制数
    #十进制数的数量
    for j in range(n_time):
        pow2 =2 ** (n_time-1-j)
        #2的幂运算
        binaries[i,j] = num10 // pow2
        num10%=pow2
#print(binaries)
#--RNN网络层循环神经网络--
class SimpleRNNLayer:
    def __init__(self,n_upper,n):
        #参数的初始值
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        #Xavier的初始值
        self.v = np.random.randn(n, n) / np.sqrt(n)
        self.b = np.zeros(n)
    #y_prev: 前一时刻的输出结
    def forward(self, x, y_prev):
        u = np.dot(x, self.w) + np.dot(y_prev, self.v) + self.b
        self.y = np.tanh(u)
        #输出
    def backward(self, x, y, y_prev, grad_y):
        delta=grad_y * (1 - y**2)
        self.grad_w += np.dot(x.T, delta)
        self.grad_v += np.dot(y_prev.T, delta)
        self.grad_b += np.sum(delta, axis=0)
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
    #--全连接输出层一
    #只有当输出层是“线性变换 + 偏置”的密集连接（Dense/FC）时才称为全连接。
    #像卷积输出层、全局池化、纯softmax(未显式线性变换)、
    #注意力输出等，都不属于全连接层
class RNNOutputLayer:
    def __init__(self,n_upper,n):
        self.w = np.random.randn(n_upper,n)/np.sqrt(n_upper)
        #Xavier的初始
        self.b = np.zeros(n)
    def forward(self,x):
        self.x =x
        u= np.dot(x, self.w) + self.b
        self.y = 1/(1+np.exp(-u)) #Sigmoid函数
    def backward(self,x, y,t):
        delta = (y-t)*y * (1-y)
        self.grad_w += np.dot(x.T,delta)
        self.grad_b += np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
    def reset_sum_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

rnn_layer = SimpleRNNLayer(n_in,n_mid)
output_layer = RNNOutputLayer(n_mid,n_out)
#--训练-一
def train(x_mb,t_mb):
    #用于保存各项输出数据的数组
    y_rnn = np.zeros((len(x_mb), n_time+1, n_mid))
    y_out = np.zeros((len(x_mb), n_time, n_out))
    y_prev =y_rnn[:,0,:]
    for i in range(n_time):
        #1，经过RNN循环网络层正向传播
        x=x_mb[:,i,:]
        rnn_layer.forward(x,y_prev)
        y= rnn_layer.y
        y_rnn[:,i+1,:]=y
        y_prev =y
        #是“序列→序列”的二进制逐位预测,需要每个时间步都产生输出,
        # 所以output_layer.forward(y)放在时间循环内(每步算一次)
        #2.经过输出层，正向传播
        output_layer.forward(y)
        y_out[:,i,:] = output_layer.y
    #反向传播
    output_layer.reset_sum_grad()
    rnn_layer.reset_sum_grad()
    grad_y=0
    for i in reversed(range(n_time)):
        #1.输出层反向传播,从最后时间序列倒序传播
        x =y_rnn[:,i+1,:]
        y = y_out[:,i,:]
        t = t_mb[:,i,:]
        output_layer.backward(x, y,t)
        grad_x_out = output_layer.grad_x
        #2.RNN层反向传播
        x=x_mb[:,i,:]
        y=y_rnn[:,i+1,:]
        y_prev = y_rnn[:,i,:]
        rnn_layer.backward(x, y, y_prev, grad_y+grad_x_out)
        grad_y = rnn_layer.grad_y_prev
    #参数的更新
    rnn_layer.update(eta)
    output_layer.update(eta)
    return y_out
#--计算误差一
def get_error(y,t):
    #误差平方用
    return 1.0/2.0*np.sum(np.square(y-t))
for i in range(n_learn):
#--随机的十进制数--
    num1 = np.random.randint(max_num//2)
    num2 = np.random.randint(max_num//2)
    #--准备输入数据-
    x1=binaries[num1]
    x2=binaries[num2]
    x_in=np.zeros((1,n_time,n_in))#(1,8,2)
    x_in[0,:,0]=x1
    x_in[0,:,1]=x2
    #将低位数放到更早的时刻，axis=1表示沿数组的第2个维度进行翻转。
    x_in =np.flip(x_in,axis=1)
    #--准备正确答案数据-
    t=binaries[num1+num2]
    t_in =t.reshape(1,n_time,n_out) #t_in变成三维ndarray,形状是(1, 8, 1)
    t_in = np.flip(t_in, axis=1)#t_in在第2维(时间步维度)上倒序
    #--训练-一
    y_out = train(x_in, t_in)
    y= np.flip(y_out,axis=1).reshape(-1)
    #把数组展平成一维向量，长度由元素总数自动推断。这里把翻转后的 y_out 
    # 变成形状 (n_time,) 的一维数组，便于后面按位转成十进制。
    #--计算误差-
    error = get_error(y_out,t_in)
    #--显示处理进度--
    if i%interval ==0:
        # 因为y是经过Sigmoid的连续输出(0~1),
        # 这里用0.5作为阈值把它二值化成0/1,便于当作二进制位进行后续计算与显示.
        y2=np.where(y<0.5,0,1)
        #二进制数的结果
        y10=0
        #十进制数的结果
        for j in range(len(y)):
            pow2=2**(n_time-1-j)#2的幂运算
            y10+=y2[j] *pow2
        print("n learn:",i)
        print("error:",error)
        print("output :",y2)
        print("correct:",t)
        #（#)
        c ="(^_^)/:" if (y2==t).all() else "orz"
        print(c+str(num1)+"+"+str(num2)+"="+str(num1+num2)+"="+str(y10))
        print("------------------------")
