import numpy as np
#使用GPU的场合
import matplotlib.pyplot as plt
from sklearn import datasets
#--设置各项参数
#图像的高度和宽度
img_size =8
#数据数量
n_noise =16
#学习系数
eta=0.001
#学习次数
n_learn=10001
#显示处理进度的间隔
interval=1000
batch_size =32
#--训练数据--
digits_data = datasets.load_digits()
x_train =np.asarray(digits_data.data)
x_train=x_train / 15*2-1
#范围为-1~1
t_train = digits_data.target
#--全连接层的父类-
class BaseLayer:
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
    #--中间层-
class MiddleLayer(BaseLayer):
    def __init__(self,n_upper,n):
        #He的初始值
        self.w = np.random.randn(n_upper,n) * np.sqrt(2/n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x=x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0,0, self.u) #ReLU
    def backward(self,grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
#一-生成器的输出层一
class GenOutLayer(BaseLayer):
    def __init__(self,n_upper,n):
        #Xavier的初始值
        self.w = np.random.randn(n_upper,n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x =x
        u = np.dot(x, self.w) + self.b
        #tanh
        self.y = np.tanh(u)
    def backward(self, grad_y):
        delta = grad_y * (1- self.y**2)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
        #--识别器的输出层
class DiscOutLayer(BaseLayer):
    def __init__(self,n_upper,n):
        #Xavier的初始值
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    def forward(self,x):
        self.x =x
        u = np.dot(x, self.w) + self.b
        #Sigmo
        self.y = 1/(1+np.exp(-u))
    def backward(self,t):
        delta = self.y-t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
#一-各网络层的初始化
gen_layers = [MiddleLayer(n_noise,32),MiddleLayer(32,64),GenOutLayer(64,img_size*img_size)]
disc_layers = [MiddleLayer(img_size*img_size, 64),MiddleLayer(64,32),DiscOutLayer(32,1)]
#--正向传播--
def forward_propagation(x,layers):
    for layer in layers:
        layer.forward(x)
        x=layer.y
    return x
#--反向传播--
def backpropagation(t,layers):
    grad_y =t
    for layer in reversed(layers):
        layer.backward(grad_y)
        grad_y = layer.grad_x
    return grad_y
#--参数的更新一一
def update_params(layers):
    for layer in layers:
        layer.update(eta)

        #--计算误差-
def get_error(y,t):
    eps = 1e-7
    #返回二值交叉误差
    return -np.sum(t*np.log(y+eps) +(1-t)*np.log(1-y+eps))/len(y)
#--计算准确率-
def get_accuracy(y,t):
    correct = np.sum(np.where(y<0.5,0,1)==t)
    return correct / len(y)
#--训练模型
def train_model(x,t,prop_layers,update_layers):
    y = forward_propagation(x,prop_layers)
    backpropagation(t,prop_layers)
    update_params(update_layers)
    return (get_error(y, t),get_accuracy(y, t))
#--生成并显示图像
def generate_images(i):
    #图像的生成
    n_rows = 16
    #行数，列数，
    n_cols=16
    #生成标准正态分布的随机噪声矩阵
    noise = np.random.normal(0, 1, (n_rows*n_cols, n_noise))
    g_imgs = forward_propagation(noise, gen_layers)
    g_imgs =g_imgs/2 +0.5 #这是把生成器输出从 [-1,1] 线性映射到 [0,1]
    #指定范围为0~1
    img_size_spaced = img_size + 2
    #完整的图像
    matrix_image = np.zeros((img_size_spaced*n_rows,img_size_spaced*n_cols))
    #将生成后的图像排列成一幅图像
    for r in range(n_rows):
        for c in range(n_cols):
            g_img = g_imgs[r*n_cols + c].reshape(img_size,img_size)
            top = r*img_size_spaced
            left = c*img_size_spaced
            matrix_image[top : top+img_size,left : left+img_size] = g_img
    plt.figure(figsize=(8,8))
    plt.imshow(matrix_image.tolist(),cmap="Greys_r")
    #删除坐标轴刻度的标签和线条
    plt.tick_params(labelbottom=False, labelleft=False,bottom=False,left=False)
    plt.show()
#--GAN的学习-
batch_half = batch_size //2
error_record = np.zeros((n_learn, 2))# 为什么是2，真假样本的损失分别计算
acc_record = np.zeros((n_learn, 2))
#这是 GAN 训练主循环：
#每轮先用噪声生成假图，训练判别器把它判为 0（假）。
#再从真实数据采样训练判别器把它判为 1（真）。
#然后固定判别器，用“生成器+判别器”联合前向/反向，只更新生成器，让生成器骗过判别器（目标为 1）。
#每隔 interval 打印误差/准确率并调用 generate_images() 可视化生成结果。
for i in range(n_learn):
    #从数据生成图像训练识别器
    noise = np.random.normal(0,1,(batch_half,n_noise))
    imgs_fake = forward_propagation(noise,gen_layers)#图像的生成
    t = np.zeros((batch_half, 1))
    error, accuracy = train_model(imgs_fake,t, disc_layers, disc_layers)
    error_record[i][0]= error
    acc_record[i][0]= accuracy
    #error 是判别器对假样本的二值交叉熵损失，accuracy 是判别器在这批假样本上的分类准确率（把输出 < 0.5 判为 0）
    #在这次调用中目标 t 全是 0，所以准确率衡量“把假图判为假”的比例。
    #使用真实图像训练识别器
    rand_ids = np.random.randint(len(x_train), size=batch_half)
    imgs_real = x_train[rand_ids,:]
    t= np.ones((batch_half,1))
    #正确答案为
    errr, accuracy = train_model(imgs_real, t,disc_layers, disc_layers)

    error_record[i][1]= error
    acc_record[i][1]= accuracy
    
    #基于合并后的模型训练生成器
    noise = np.random.normal(0,1,(batch_size,n_noise))
    #把目标 t 设为全 1，表示希望判别器把生成器输出判为“真”。
    t = np.ones((batch_size, 1))
    #正确答案为，：前向/反向通过生成器+判别器，但只更新生成器参数，从而让生成器学会“骗过判别器”。
    #通过把目标 t 设为全 1，让判别器输出尽量接近“真”，并且只更新生成器参数、冻结判别器。
    #这样生成器被梯度驱动去生成更像真实的数据，从而“骗过”判别器。
    train_model(noise,t, gen_layers+disc_layers,gen_layers)
    #只训练生成
    #以固定间隔显示误差和生成后的图像
    if i %interval == 0:
        print ("n_learn:",i)
        print ("Error_fake:",error_record[i][0],"Acc_fake:", acc_record[i][0])
        print ("Error_real:", error_record[i][1],"Acc_real:", acc_record[i][1])
        generate_images(i)
