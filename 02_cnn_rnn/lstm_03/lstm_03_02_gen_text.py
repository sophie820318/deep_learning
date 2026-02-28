import numpy as np
# import cupy as np
#使用GPU环境
import matplotlib.pyplot as plt
#--各项设置常数一
n_time = 20
#时间序列数据的数量
n_mid=128
#中间层的神经元数量
eta=0.01
#学习系数
clip_const = 0.02
#用于决定范数的最大值常量
beta=2
#概率分布的宽度（在确定下
epoch= 50
batch_size = 128
def sigmoid(x):
    return 1/(1+np.exp(-x))
def clip_grad(grads,max_norm):
    norm = np.sqrt(np.sum(grads*grads))
    r= max_norm / norm
    if r<1:
        clipped_grads = grads *r
    else:
        clipped_grads = grads
    return clipped_grads

with open("kaijin20.txt", mode="r", encoding="utf-8") as f:
#文件的读入
    text = f.read()
#使用len函数获取字符串长度
print("字数：", len(text))
#-文字与索引1的关联-
chars_list=sorted(list(set(text)))#使用set函数消除重复文字
n_chars=len(chars_list)
print("字数（无重复）：", n_chars)
char_to_index = {}
#文字作为键值、索引作为数据
index_to_char = {}
#索引作为键值、文字作为数据
for i,char in enumerate(chars_list):
    char_to_index[char] = i
    index_to_char[i] = char
#一按时间序列排列的文字及其下一个文字一-
seq_chars=[]
next_chars=[]
for i in range(0, len(text) - n_time):
    seq_chars.append(text[i:i + n_time])
    next_chars.append(text[i + n_time])
#--使用独热格式表示输入数据和正确答案数据-
input_data = np.zeros((len(seq_chars), n_time, n_chars), dtype=np.bool)
correct_data = np.zeros((len(seq_chars), n_chars), dtype=np.bool)
for i, chars in enumerate(seq_chars):
    #使用独热格式表示正确答案数据
    correct_data[i, char_to_index[next_chars[i]]] = 1
    for j, char in enumerate(chars):
        input_data[i,j, char_to_index[char]] =1
#--LSTM网络层-
class LSTMLayer:
    def __init__(self, n_upper, n):
        #各项参数的初始值
        self.w = np.random.randn(4, n_upper, n) / np.sqrt(n_upper)
        self.v = np.random.randn(4, n, n) / np.sqrt(n)
        self.b = np.zeros((4, n))
        #y_prev，c_prev:前一时刻的输出和记忆单元
    def forward(self,x,y_prev, c_prev):
        u = np.matmul(x, self.w) + np.matmul(y_prev, self.v) + self.b.reshape(4,1,-1)
        a0=sigmoid(u[0])#忘记门
        a1 =sigmoid(u[1])#输入门
        a2=np.tanh(u[2])#新的记忆
        a3=sigmoid(u[3])#输出门
        self.gates=np.stack((a0,a1,a2,a3))
        self.c=a0*c_prev + a1*a2#记忆单元
        self.y=a3 *np.tanh(self.c)#输出
    def backward(self,x,y,c,y_prev,c_prev,gates,grad_y,grad_c,):
        a0,a1,a2,a3 =gates
        tanh_c= np.tanh(c)
        r=grad_c+(grad_y*a3)*(1-tanh_c**2)
        #各个delta
        delta_a0=r*c_prev*a0*(1-a0)
        delta_a1=r*a2*a1*(1-a1)
        delta_a2=r*a1*(1-a2**2)
        delta_a3=grad_y*tanh_c*a3 *(1-a3)
        deltas=np.stack((delta_a0,delta_a1,delta_a2,delta_a3))
        #各项常数的梯度
        self.grad_w += np.matmul(x.T, deltas)
        self.grad_v += np.matmul(y_prev.T,deltas)
        self.grad_b += np.sum(deltas, axis=1)
        #×的梯度
        grad_x= np.matmul(deltas, self.w.transpose(0,2,1))
        self.grad_x= np.sum(grad_x, axis=0)
        #y_prev的梯度
        grad_y_prev = np.matmul(deltas, self.v.transpose(0, 2, 1))
        self.grad_y_prev = np.sum(grad_y_prev, axis=0)
        #c_prev的梯度
        self.grad_c_prev =r * a0
    def reset_sum_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.v -= eta * self.grad_v
        self.b -= eta * self.grad_b
    def clip_grads(self, clip_const):
        self.grad_w = clip_grad(self.grad_w, clip_const+np.sqrt(self.grad_w.size))
        self.grad_v = clip_grad(self.grad_v, clip_const+np.sqrt(self.grad_v.size))

class OutputLayer:
    def __init__(self,n_upper,n):
        #Xavier的初始值
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros((n))
    def forward(self,x):
        self.x = x
        u= np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u),axis=1).reshape(-1,1)
        #Softmax函
    def backward(self,t):
        delta=self.y-t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta,axis=0)
        self.grad_x = np.dot(delta, self.w.T)
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
lstm_layer = LSTMLayer(n_chars, n_mid)
output_layer = OutputLayer(n_mid, n_chars)
#--训练-一
def train(x_mb, t_mb):
#正向传播LSTM层
    y_rnn = np.zeros((len(x_mb), n_time+1, n_mid))
    c_rnn = np.zeros((len(x_mb), n_time+1, n_mid))
    gates_rnn = np.zeros((4, len(x_mb), n_time, n_mid))
    y_prev = y_rnn[:,0,:]
    c_prev = c_rnn[:,0,:]
    for i in range(n_time):
        x = x_mb[:, i, :]
        lstm_layer.forward(x, y_prev, c_prev)
        y = lstm_layer.y
        y_rnn[:, i+1, :] = y
        y_prev = y

        c = lstm_layer.c
        c_rnn[:, i+1, :] = c
        c_prev = c
        gates = lstm_layer.gates
        gates_rnn[:,:, i,:] = gates
        #正向传播输出层
    output_layer.forward(y)
    #反向传播输出层
    output_layer.backward(t_mb)
    grad_y = output_layer.grad_x
    grad_c = np.zeros_like(lstm_layer.c)
    #反向传播LSTM层
    lstm_layer.reset_sum_grad()
    for i in reversed(range(n_time)):
        x=x_mb[:,i,:]
        y= y_rnn[:, i+1,:]
        c=c_rnn[:,i+1,:]
        y_prev = y_rnn[:, i,:]
        c_prev = c_rnn[:,i,:]
        gates = gates_rnn[:, :, i,:]
        lstm_layer.backward(x, y, c, y_prev, c_prev, gates, grad_y, grad_c)
        grad_y=lstm_layer.grad_y_prev
        grad_c=lstm_layer.grad_c_prev
    #参数的更新
    lstm_layer.clip_grads(clip_const)
    lstm_layer.update(eta)
    output_layer.update(eta)
def predict(x_mb):
    #正向传播LSTM层
    y_prev = np.zeros((len(x_mb), n_mid))
    c_prev = np.zeros((len(x_mb), n_mid))
    for i in range(n_time):
        x=x_mb[:, i,:]
        lstm_layer.forward(x, y_prev, c_prev)
        y= lstm_layer.y
        y_prev = y
        c=lstm_layer.c
        c_prev =c
    #正向传播输出层
    output_layer.forward(y)
    return output_layer.y
    #--计算误差-
def get_error(x,t):
    limit = 1000
    if len(x) >limit:
        #设置测算的样本数量上限
        index_random = np.arange(len(x))
        np.random.shuffle(index_random)
        x = x[index_random[:limit],:]
        t = t[index_random[:limit],:]
    y=predict(x)
    return -np.sum(t * np.log(y + 1e-7))//batch_size
    # azsy((-odunsduuna
def create_text():
    prev_text = text[0:n_time]
    #输入数据
    created_text = prev_text
    #生成的文本
    print("Seed:",created_text)
    for i in range(200):
        #生成200个字的文
        #将输入数据转换为独热格式
        x= np.zeros((1,n_time,n_chars))
        for j, char in enumerate(prev_text):
            x[0, j, char_to_index[char]] =1
        #进行预测，得到下一个文字
        y= predict(x)
        #概率分布的调整
        p=y[0] ** beta
        #将p的合计变为1
        p=p/np.sum(p)
        next_index =np.random.choice(len(p), size=1,p=p)

        next_char = index_to_char[int(next_index[0])]
        created_text += next_char
        prev_text = prev_text[1:] + next_char
    print(created_text)
    print() #换行
error_record =[]
n_batch=len(input_data)// batch_size
for i in range(epoch):
        #每轮epoch的批次数
        #--学习—一
        index_random=np.arange(len(input_data))
        np.random.shuffle(index_random)
        for j in range(n_batch):
            #取出小批次数据进行训练
            mb_index = index_random[j*batch_size : (j+1)*batch_size]
            x_mb = input_data[mb_index, :]
            t_mb = correct_data[mb_index, :]
            train(x_mb, t_mb)
            print("\repoch:" +str(i+1)+"/"+ str(epoch) + "  " + str(j+1) + "/" + str(n_batch), end="")
        error = get_error(input_data, correct_data)
        error_record.append(error)
        print("error:"+str(error))
        create_text()
        #--显示处理进度一
plt.plot(range(1, len(error_record)+1), error_record,label="Error")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.show()