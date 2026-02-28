import numpy as np
#import cupy as np
#使用GPU的场合
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
#--各项设置参数--
img_size =8
#图像的高度和宽度
n_time=4
#时间序列数据的数量
n_in = img_size
#输入层的神经元数量
n_mid= 128
#中间层的神经元数量
n_out =img_size
#显示图像的张数
n_disp=10
#学习系数
eta=0.01
epochs =201
batch_size =32
#显示处理进度的间隔
interval=50
def sigmoid(x):
    return 1/(1+np.exp(-x))
#--数据的准备一
digits = datasets.load_digits()
#提供GPU支持
digits = np.asarray(digits.data)
#把一维的每个样本（长度 64）重塑成 8×8图像，-1 表示样本数自动推断。

digits_imgs= digits.reshape(-1, img_size, img_size)
print("digits_imgs shape:", digits_imgs.shape)#(1797, 8, 8)
#将像素从 [0,15] 归一化到 [0,1] 便于训练。
digits_imgs /= 15#设置为0~1的范围

disp_imgs = digits_imgs[:n_disp]#前10张用于显示结果

train_imgs = digits_imgs[n_disp:]#n_disp后用于训练1787维度，(1787, 8, 8)
n_sample_in_img = img_size-n_time#8-4=4
#一张图像中的样本数量
n_sample = len(train_imgs) * n_sample_in_img#1787*4=7148
#样本数量
input_data= np.zeros((n_sample, n_time, n_in)) #输入7148*4*8
correct_data = np.zeros((n_sample,n_out))#8
for i in range(len(train_imgs)):#1787
    #正确答案
    for j in range(n_sample_in_img):#4
        sample_id = i*n_sample_in_img + j
        #用紧接着的下一行作为监督目标。这样模型学到从上半部分行生成下一行的规律，
        # 最终可逐行生成整张 8×8 图像。
        input_data[sample_id] = train_imgs[i, j:j+n_time]
        correct_data[sample_id] = train_imgs[i, j+n_time]
        #--分割为训练数据和测试数据-一
x_train,x_test, t_train, t_test = train_test_split(input_data, correct_data)
#--GRU层--
class GRULayer:
    def __init__(self,n_upper,n):
        #参数的初始值
        self.w= np.random.randn(3,n_upper,n) / np.sqrt(n_upper)
        #Xavier的初始值
        self.v = np.random.randn(3, n, n) / np.sqrt(n)
    def forward(self, x, y_prev):
        a0 = sigmoid(np.dot(x, self.w[0]) + np.dot(y_prev, self.v[0]))
        a1 = sigmoid(np.dot(x, self.w[1]) + np.dot(y_prev, self.v[1]))
        #更新门
        a2=np.tanh(np.dot(x,self.w[2])+np.dot(a1*y_prev,self.v[2]))
        
        #新的记忆
        self.gates = np.stack((a0, a1, a2))
        self.y=(1-a0)*y_prev + a0*a2
        #输出
    def backward(self,x,y,y_prev,gates,grad_y):
        a0,a1,a2 =gates
        #新的记忆
        delta_a2=grad_y *a0*(1-a2**2)
        self.grad_w[2] += np.dot(x.T,delta_a2)
        self.grad_v[2] += np.dot((a1*y_prev).T,delta_a2)
        #更新门
        delta_a0 = grad_y * (a2 - y_prev) * a0 * (1 - a0)
        self.grad_w[0] += np.dot(x.T, delta_a0)
        self.grad_v[0] += np.dot(y_prev.T, delta_a0)
        #复位门
        s = np.dot(delta_a2,self.v[2].T)
        delta_a1 = s * y_prev * a1 * (1 - a1)
        self.grad_w[1] += np.dot(x.T,delta_a1)
        self.grad_v[1] += np.dot(y_prev.T,delta_a1)
        #×的梯度
        self.grad_x=np.dot(delta_a0, self.w[0].T)+ np.dot(delta_a1,self.w[1].T)+np.dot(delta_a2,self.w[2].T)
        #y_prev的梯度
        self.grad_y_prev = np.dot(delta_a0, self.v[0].T)+ np.dot(delta_a1,self.v[1].T)+a1*s+ grad_y*(1-a0)
    def reset_sum_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.v -= eta * self.grad_v
        #--全连接输出层-—
class OutputLayer:
    def __init__(self,n_upper,n):
        self.w = np.random.randn(n_upper, n) /np.sqrt(n_upper)
        #Xavier的初始值
        self.b = np.zeros(n)
    def forward(self,x):
        self.x=x
        u = np.dot(x,self.w) + self.b
        self.y=1/(1+np.exp(-u))
    #Sigmoid函数
    def backward(self, t):
        delta =(self.y-t) * self.y *(1-self.y)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
#--各网络层的初始化一一
gru_layer = GRULayer(n_in,n_mid)
output_layer = OutputLayer(n_mid,n_out)
#--训练-
def train(x_mb,t_mb):
    #正向传播GRU层
    y_rnn = np.zeros((len(x_mb),n_time+1, n_mid))
    gates_rnn = np.zeros((3, len(x_mb), n_time, n_mid))
    y_prev =y_rnn[:,0,:]
    for i in range(n_time):
        x=x_mb[:,i,:]
        gru_layer.forward(x,y_prev)
        y = gru_layer.y
        y_rnn[:,i+1,:]=y
        y_prev =y
        gates = gru_layer.gates
        gates_rnn[:,:,i,:] = gates
        #正向传播输出层
    output_layer.forward(y)
    #反向传播输出层
    output_layer.backward(t_mb)
    grad_y = output_layer.grad_x
    gru_layer.reset_sum_grad()
    for i in reversed(range(n_time)):
        x = x_mb[:, i, :]
        y = y_rnn[:, i+1, :]
        y_prev = y_rnn[:, i, :]
        gates = gates_rnn[:, :, i, :]
        gru_layer.backward(x,y,y_prev, gates, grad_y)
        grad_y = gru_layer.grad_y_prev
    #参数的更新
    gru_layer.update(eta)
    output_layer.update(eta)
    #--预测—一
def predict(x_mb):
    #正向传播GRU层
    y_prev = np.zeros((len(x_mb),n_mid))
    for i in range(n_time):
        x = x_mb[:, i, :]
        gru_layer.forward(x,y_prev)
        y = gru_layer.y
        y_prev = y
    #正向传播输出层
    output_layer.forward(y)
    return output_layer.y
#--计算误差--
def get_error(x,t):
    y=predict(x)
    return np.sum(np.square(y-t)) / len(x)
    #误差
    #--生成并显示图像
def generate_images():
    #原始图像
    plt.figure(figsize=(10,1))
    for i in range(n_disp):
        ax = plt.subplot(1,n_disp,i+1)
        plt.imshow(disp_imgs[i].tolist(),cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #不显
    plt.show()
    #下半部分是基于RNN生成的图像
    gen_imgs=disp_imgs.copy()
    plt.figure(figsize=(10,1))
    for i in range(n_disp):
        for j in range(n_sample_in_img):
            x= gen_imgs[i, j: j+n_time].reshape(1,n_time,img_size)
            gen_imgs[i, j+n_time] =predict(x)[0]
        ax = plt.subplot(1,n_disp,i+1)
        plt.imshow(gen_imgs[i].tolist(),cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
n_batch=len(x_train)// batch_size
#每轮epoch的批次数
for i in range(epochs):
    #--学习-一
    index_random = np.arange(len(x_train))
    np.random.shuffle(index_random)
    #将索引的顺序打乱
    for j in range(n_batch):
        #提取小批次
        mb_index = index_random[j*batch_size :(j+1)*batch_size]
        x_mb =x_train[mb_index,:]
        t_mb = t_train[mb_index,:]
        #训练
        train(x_mb,t_mb)
    #--显示处理进度-一
    if i%interval == 0:
        #测量误差
        error_train = get_error(x_train, t_train)
        error_test = get_error(x_test,t_test)
        print("Epoch:"+ str(i)+"/"+ str(epochs-1),"Error_train:"+ str(error_train),
              "Error_test:"+ str(error_test))
        #图像的生成
        generate_images()
      