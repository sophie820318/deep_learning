import numpy as np
#使用GPU的场合
#import cupy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#--各项设置参数一
img_size =8
#图像的高度和宽度，#输入层和输出层的神经元数量，8*8=64
n_in_out = img_size * img_size
#中间层的神经元数量
n_mid=16
#隐藏层，潜空间变量
n_z=2
eta=0.001
#学习系数
epochs = 201
batch_size = 32
interval =20
#显示处理进度的间隔
#一一训练数据一
digits_data = datasets.load_digits()
x_train = np.asarray(digits_data.data)
#设置为0~1的范围，归一化
x_train /=15
#是每张手写数字图片的类别标签（0–9），
# 对应 digits_data.data 中每一行样本的真实数字
t_train = digits_data.target
#--全连接层的父类一一
class BaseLayer:
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
    #--中间层一一
class MiddleLayer(BaseLayer):
    def __init__(self,n_upper,n):
        #He的初始值
        self.w = np.random.randn(n_upper, n) * np.sqrt(2/n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x =x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u) #ReLU
    def backward(self,grad_y):
        delta = grad_y * np.where(self.u <= 0,0,1)
        self.grad_w = np.dot(self.x.T,delta)
        self.grad_b = np.sum(delta,axis=0)
        self.grad_x = np.dot(delta, self.w.T)
class ParamsLayer(BaseLayer):
    def __init__(self,n_upper,n):
        #Xavier的初始值
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b= np.zeros(n)
    def forward(self,x):
        self.x=x
        self.u = np.dot(x, self.w) + self.b
        #恒等函数
        self.y =self.u
    def backward(self,grad_y):
        #delta 表示该层“误差信号”，即损失对该层线性输入的敏感度  
        #衡量此层输出对总损失的影响。
        delta= grad_y
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
#输出层-一
class OutputLayer(BaseLayer):
    def __init__(self,n_upper,n):
        #Xavier的初始值
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x=x
        u= np.dot(x, self.w) + self.b
        self.y=1/(1+np.exp(-u))
    #Sigmoid函数
    def backward(self,t):
        delta = self.y -t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
class LatentLayer:
    def forward(self,mu, log_var):
        self.mu = mu #平均值
        self.log_var = log_var #方差的对数
        self.epsilon = np.random.randn(*log_var.shape)
        #z=μ+ϵ⋅exp(1/2 * logσ^2)
        self.z = mu + self.epsilon*np.exp(log_var/2)
    def backward(self, grad_z):
        self.grad_mu = grad_z + self.mu
        self.grad_log_var = grad_z*self.epsilon/2*np.exp(
        self.log_var/2)-0.5*(1-np.exp(self.log_var))

#--各网络层的初始化
middle_layer_enc = MiddleLayer(n_in_out,n_mid)
mu_layer = ParamsLayer(n_mid,n_z)
log_var_layer = ParamsLayer(n_mid,n_z)
z_layer = LatentLayer()
#Decoder
middle_layer_dec = MiddleLayer(n_z,n_mid)
output_layer = OutputLayer(n_mid, n_in_out)
#--正向传播
def forward_propagation(x_mb):
    #Encoder
    middle_layer_enc.forward(x_mb)
    mu_layer.forward(middle_layer_enc.y)
    log_var_layer.forward(middle_layer_enc.y)
    z_layer.forward(mu_layer.y, log_var_layer.y)
    #Decoder
    middle_layer_dec.forward(z_layer.z)
    output_layer.forward(middle_layer_dec.y)
#--反向传播-一
def backpropagation(t_mb):
    #Decoder
    output_layer.backward(t_mb)
    middle_layer_dec.backward(output_layer.grad_x)
    #Encoder
    z_layer.backward(middle_layer_dec.grad_x)
    log_var_layer.backward(z_layer.grad_log_var)
    mu_layer.backward(z_layer.grad_mu)
    middle_layer_enc.backward(mu_layer.grad_x + log_var_layer.grad_x)
def update_params():
    middle_layer_enc.update(eta)
    mu_layer.update(eta)
    log_var_layer.update(eta)
    middle_layer_dec.update(eta)
    output_layer.update(eta)
    #--计算误差--
def get_rec_error(y, t):
    eps=1e-7
    return -np.sum(t*np.log(y+eps) +(1-t)*np.log(1-y+eps))/ len(y)
def get_reg_error(mu,log_var):
    return -np.sum(1 + log_var -mu**2-np.exp(log_var))/ len(mu)
rec_error_record =[]
reg_error_record =[]
total_error_record = []
#每轮epo
n_batch =len(x_train) // batch_size
for i in range(epochs):
    #--学习一
    index_random = np.arange(len(x_train))
    np.random.shuffle(index_random)
    #对索引进行
    for j in range(n_batch):
        #提取小批次数据
        mb_index = index_random[j*batch_size:(j+1)*batch_size]
        x_mb = x_train[mb_index,:]
        #正向传播与反向传播
        forward_propagation(x_mb)
        backpropagation(x_mb)
        #权重与偏置的更新
        update_params()
    #--求取误差
    forward_propagation(x_train)
    rec_error = get_rec_error(output_layer.y, x_train)
    reg_error = get_reg_error(mu_layer.y, log_var_layer.y)
    total_error = rec_error + reg_error
    rec_error_record.append(rec_error)
    reg_error_record.append(reg_error)
    total_error_record.append(total_error)
    #一-显示处理进度一
    if i%interval == 0:
        print("Epoch:",i,"Rec_error:",rec_error,
        "Reg_error",reg_error,
        "Total_error",total_error)
plt.plot(range(1,len(rec_error_record)+1),rec_error_record,
          label="Rec_error")
plt.plot(range(1,len(reg_error_record)+1),reg_error_record, 
         label="Reg_error")
plt.plot(range(1,len(total_error_record)+1),total_error_record,
         label="Total_error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
