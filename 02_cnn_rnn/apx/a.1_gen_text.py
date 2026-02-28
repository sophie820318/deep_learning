import numpy as np
#使用GPU的场合
import matplotlib.pyplot as plt
#--各项设置参数-
n_time =20
#时间序列数据的数量
#中间层的神经元数量
n_mid=128
#学习系数
eta=0.01
#决定范数最大值的常量
clip_const = 0.02
#概率分布的范围,确定下一字名
beta=2
epoch = 60
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
#--训练用的文章一一
#文件的读取
with open("kaijin20.txt", mode="r", encoding="utf-8") as f:
    text = f.read()
#使用len也可以获取字符
print("字符数：", len(text))
#--文字与索引的关联
#使用set去
chars_list = sorted(list(set(text)))
n_chars = len(chars_list)
print("字符数（无重复）：", n_chars)
#字符为键、索引为值
char_to_index = {}
#索引为键、字符为值
index_to_char = {}
for i,char in enumerate(chars_list):
    char_to_index[char] = i
    index_to_char[i] = char
#-时间序列中排列的字符与后面的字符一
seq_chars =[]
next_chars = []
for i in range(0,len(text) - n_time):
    seq_chars.append(text[i: i + n_time])
    next_chars.append(text[i + n_time])
#--输入与正确答案的独热格式
input_data = np.zeros((len(seq_chars), n_time, n_chars), dtype=np.bool)
correct_data = np.zeros((len(seq_chars), n_chars), dtype=np.bool)
for i,chars in enumerate(seq_chars):
    #将正确答案用独热格式表示
    correct_data[i, char_to_index[next_chars[i]]] = 1
    for j,char in enumerate(chars):
        #将输入用独热格式表示
        input_data[i,j,char_to_index[char]] = 1
#--RNN层
class SimpleRNNLayer:
    def __init__(self,n_upper,n):
        #参数的初始值
        self.w = np.random.randn(n_upper,n) /np.sqrt(n_upper)
        #Xavier的初始值
        self.v =np.random.randn(n,n) /np.sqrt(n)
        #Xavier的初始值
        self.b = np.zeros(n)
    def forward(self,x,y_prev):
        #y_prev:前一时刻的输出
        u=np.dot(x, self.w) +np.dot(y_prev, self.v) + self.b
        self.y = np.tanh(u)
    #输出
    def backward(self,x,y,y_prev, grad_y):
        delta = grad_y * (1 - y**2)
        #各个梯度
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
    def clip_grads(self,clip_const):
        self.grad_w = clip_grad(self.grad_w,
        clip_const*np.sqrt(self.grad_w.size))
        self.grad_v = clip_grad(self.grad_v,
        clip_const*np.sqrt(self.grad_v.size))
#--全连接输出层一
class OutputLayer:
    def __init__(self,n_upper,n):
        self.w = np.random.randn(n_upper,n) / np.sqrt(n_upper)
        #Xavier的初始值
        self.b = np.zeros(n)
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u), axis=1).reshape(-1,1)
    #Softmax函数
    def backward(self,t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
#--各网络层的初始化--
rnn_layer = SimpleRNNLayer(n_chars,n_mid)
output_layer = OutputLayer(n_mid,n_chars)
#--训练-一
def train(x_mb,t_mb):
    y_rnn = np.zeros((len(x_mb), n_time+1, n_mid))
    y_prev=y_rnn[:,0,:]
    for i in range(n_time):
        x=x_mb[:,i,:]
        rnn_layer.forward(x,y_prev)
        y= rnn_layer.y
        y_rnn[:,i+1,:] =y
        y_prev =y
    #正向传播输出层
    output_layer.forward(y)
    #反向传播输出层
    output_layer.backward(t_mb)
    grad_y = output_layer.grad_x
    #反向传播RNN层
    rnn_layer.reset_sum_grad()
    for i in reversed(range(n_time)):
        x=x_mb[:,i,:]
        y=y_rnn[:,i+1,:]
        y_prev = y_rnn[:,i,:]
        rnn_layer.backward(x,y,y_prev,grad_y)
        grad_y = rnn_layer.grad_y_prev
    rnn_layer.clip_grads(clip_const)
    rnn_layer.update(eta)
    output_layer.update(eta)
#-预测
def predict(x_mb):
    #正向传播RNN层
    y_prev = np.zeros((len(x_mb), n_mid))
    for i in range(n_time):
        x=x_mb[:,i,:]
        rnn_layer.forward(x,y_prev)
        y = rnn_layer.y
        y_prev=y
    #正向传播输出层
    output_layer.forward(y)
    return output_layer.y
    #--计算误差-一
def get_error(x,t):
    limit =1000
    if len(x) > limit:
        #设置测定样本数量的！
        index_random = np.arange(len(x))
        np.random.shuffle(index_random)
        x = x[index_random[:limit],:]
        t = t[index_random[:limit],:]
    y= predict(x)
    #交叉熵误差
    return -np.sum(t*np.log(y+1e-7))/batch_size
def create_text():
    prev_text = text[0:n_time]
    #输入
    created_text = prev_text
    #生成的文本
    print("Seed:",created_text)
    for i in range(200):
        #生成200字符的文章
        x= np.zeros((1, n_time, n_chars))# 将输入设置为独热编码
        for j,char in enumerate(prev_text):
            x[0,j,char_to_index[char]] =1
            #进行预测，以获取下一个字符
            y=predict(x)
            p=y[0] **beta
            #调整概率分布
            p=p/np.sum(p)
            #将p的合计值设置为1
            next_index = np.random.choice(len(p), size=1, p=p)
            next_char = index_to_char[int(next_index[0])]
            created_text += next_char
            prev_text = prev_text[1:] + next_char
    print(created_text)
    print()
error_record =[]
#每轮epoch的批次数
n_batch =len(input_data)// batch_size
for i in range(epoch):
    index_random=np.arange(len(input_data))
    #打乱索引的顺序
    np.random.shuffle(index_random)
    for j in range(n_batch):
        #提取小批次
        mb_index = index_random[j*batch_size:(j+1)*batch_size]
        x_mb = input_data[mb_index,:]
        t_mb = correct_data[mb_index,:]
        train(x_mb,t_mb)
        #一-显示处理进度--
        print("\rEpoch:"+str(i+1)+"/"+str(epoch)+" "+str(j+1)+"/"+str(n_batch),end="")
    #-求取误差一一
    error = get_error(input_data,correct_data)
    error_record.append(error)
    print("Error:"+str(error))
    #--显示处理进度一
    create_text()
plt.plot(range(1, len(error_record)+1), error_record, label="error")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.show()