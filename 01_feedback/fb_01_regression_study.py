#page 1
# %matplotlib inline  # Jupyter魔法命令，已注释/移除
import numpy as np
import matplotlib.pyplot as plt
#--准备输入和正确答案数据2
#将输入收敛到-1.0~1.0的范围之内
# input_data = np.arange(0,np.pi*2,0.1)
# correct_data = np.sin(input_data)
# #归一化
# input_data = (input_data -np.pi)/np.pi
# n_data = len(correct_data)
input_data= np.arange(0,np.pi*2,0.1)
correct_data = np.sin(input_data)

#Normalization:(-1~1)
input_data = (input_data -np.pi)/np.pi
n_data = len(correct_data)

n_in=1   #输入层的神经元数量
n_mid=3  #中间层的神经元数量
n_out=1  #输出层的神经元数量


wb_width =0.01 #权重和偏置的扩散程度
eta = 0.1  #学习系数
epoch = 2001
interval = 200  #显示进度的间隔

#--中间层--
#初始化设置
# class MiddleLayer:
#     def __init__(self,n_upper,n):
#         #权重矩阵
#         self.w = wb_width *np.random.randn(n_upper, n)
#         #偏置（向量）
#         self.b =wb_width *np.random.randn(n)
#     def forward(self,x):
#         self.x=x
#         u =np.dot(x, self.w) + self.b
#         #sigmoid函数 激活函数，将 u 映射到 (0, 1)，Logistic 函数
#         self.y=1/(1+np.exp(-u))
    
#     def backward(self,grad_y):
#         delta = grad_y *(1-self.y)*self.y  #sigmoid函数的微分
#         self.grad_w=np.dot(self.x.T,delta)
#         self.grad_b=np.sum(delta,axis=0)
#         self.grad_x=np.dot(delta,self.w.T)
#     def update(self,eta):
#         #权重和偏置的更新
#         self.w-=eta*self.grad_w
#         self.b-=eta*self.grad_b


class MiddleLayer:
    def __init__(self,n_upper,n):
        #返回形状为 (n_upper, n) 的二维 NumPy 数组，
        #数组中包含从标准正态分布（均值为 0，标准差为 1）中采样的浮点数。
        self.w = wb_width*np.random.randn(n_upper,n) 
        self.b = wb_width*np.random.randn(n) #多少个神经元就有多少个偏置，偏置是向量
    def forward(self,x):
        self.x = x
        u = np.dot(x,self.w)+ self.b
        #引入非线性,使网络能拟合复杂函数,可直接表示某类的概率,Sigmoid 的输出范围(0，1),便于网络训练
        #也容易出现梯度消失,在反向传播时,梯度通过链式法则逐层相乘,每一层的偏导数都是Sigmoid导数σ‘(x)<=0.25
        self.y = 1/(1+np.exp(-u))
    def backward(self,grad_y): # 
        #δ= ∂E/∂u = ∂E/∂y * ∂y/∂u = grad_y *(激励函数导数)
        # d(g(u))^{-1}/du = -(g(u))^{-2} *g'(u)=
        #self.y = 1/(1+np.exp(-u))
        delta = grad_y* (1-self.y)*self.y #sigmoid函数的微分
        #∂E/∂w =∂E/∂u * ∂u/∂w = δ * x = 写成矩阵形式：(x.T . δ)
        self.grad_w = np.dot(self.x.T,delta)
        #∂E/∂b =∂E/∂u * ∂u/∂b = δ * 1 = δ的各行求和
        self.grad_b = np.sum(delta,axis=0)
        ##∂E/∂x =∂E/∂u * ∂u/∂x = δ * w= (δ . w.T):写成矩阵形式
        self.grad_x = np.dot(delta,self.w.T)

class OutputLayer:
    def __init__(self,n_upper,n):
        #初始化设置
        self.w=wb_width *np.random.randn(n_upper,n)
        #权重（矩阵）
        self.b =wb_width *np.random.randn(n)
    #偏置（向量）
    def forward(self,x):
        #正向传播
        self.x=x
        u =np.dot(x,self.w) + self.b
        #恒等函数
        self.y=u
        #反向传播
    def backward(self,t):
        delta =self.y-t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b=np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)
    #权重和偏置的更
    def update(self,eta):
        self.w -= eta * self.grad_w
        self.b -=eta *self.grad_b
#一-各个网络层的初始化
middle_layer = MiddleLayer(n_in,n_mid)  #1,3,1
output_layer = OutputLayer(n_mid,n_out)
#学习
for i in range(epoch):
    #随机打乱索引值
    index_random =np.arange(n_data)#返回一个从0到n_data-1的整数数组
    np.random.shuffle(index_random)
    #用于结果集的显示
    total_error=0
    plot_x=[]
    plot_y=[]
    for idx in index_random:
        #输入
        x=input_data[idx:idx+1]
        #正确答案
        t=correct_data[idx:idx+1]
        #正向传播
        #将输入转换为矩阵
        middle_layer.forward(x.reshape(1, 1))
        output_layer.forward(middle_layer.y)
        #反向传播
        #将正确答案转换为矩阵
        output_layer.backward(t.reshape(1, 1))
        middle_layer.backward(output_layer.grad_x)
        #权重和偏置的更新
        middle_layer.update(eta)
        output_layer.update(eta)
        if i%interval ==0:
            y=output_layer.y.reshape(-1)  #将矩阵还原成向量
            #误差的计算
            total_error +=1.0/2.0*np.sum(np.square(y- t))  #平方和误差
            #输出的记录
            plot_x.append(x)
            plot_y.append(y)
    if i%interval ==0:
        #用图表显示输出
        plt.plot(input_data,correct_data,linestyle="dashed")
        plt.scatter(plot_x,plot_y,marker="x")
        plt.show()
        #显示epoch次数和误差
        print("Epoch:"+ str(i)+"/"+ str(epoch),
        "Loss:"+str(total_error/n_data))