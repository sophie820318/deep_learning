#page 1
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#-坐标—
X=np.arange(-1.0,1.1,0.1)    #numpy.ndarray  X=[x0,x1,x2,...,x20],X是长度为21的一维向量
Y=np.arange(-1.0,1.1,0.1)    #numpy.ndarray  Y=[y0,y1,y2,...,y20],Y是长度为21的一维向量
#--创建输入和正确答案数据
input_data = []  #定义为一个空的 Python 列表（list）,input_data =[]
correct_data = []
for x in X:
    for y in Y:
        input_data.append([x, y])
        if y< np.sin(np.pi*x):     #如果y坐标在正弦曲线下
            correct_data.append([0,1])   #下方的区域
        else:
            correct_data.append([1,0])   #上方的区域
#input_data 和 correct_data 被X和Y的每个元素排列组合填满数据,input_data =[[x0,y0],[x0,y1],[x0,y2],...[x20,y20]] 
n_data= len(correct_data) #数据的数量 n_data=441= 21*21
print("n_data:", n_data) 
input_data=np.array(input_data) #转换为numpy.ndarray数组类型 (441,2),
                     #input_data=[[x0,y0],
                                 #[x0,y1],
                                 #[x0,y2],
                                 #[x0,y2],
                                 #...
                                 #[x20,y20]]
correct_data=np.array(correct_data)
# 数据处理的一种方式，代码生成的，由list转换为ndarray的一种处理手段
#page 2
#--各个设置值
n_in=2   #输入层的神经元数量,输入的维度有要解决问题的变量的个数决定
n_mid=6  #中间层的神经元数量，可以多也可以少20以内都可以
n_out=2  #输出层的神经元数量，输出的维度由要解决的分类问题的分类个数决定

wb_width=0.01  #权重和偏置的扩散程度，避免陷入局部最优解
eta=0.1        #学习系数
epoch=101 
interval =10  #显示进度的间隔时间

class MiddleLayer:
    def __init__(self, n_upper, n):
        #numpy.ndarray，生成服从标准正态分布(均值为0,方差为1)的随机数数组，[2,6]数组 
        self.w = wb_width * np.random.randn(n_upper, n) #权重矩阵
        self.b = wb_width * np.random.randn(n)          #偏置向量

    def forward(self,x):
        self.x=x
        u=np.dot(x,self.w) + self.b
        self.y=1/(1+np.exp(-u))  #sigmoid函数

    def backward(self,grad_y):
        delta =grad_y*(1-self.y)*self.y
        self.grad_w=np.dot(self.x.T,delta)
        self.grad_b=np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)

    def update(self,eta):
        self.w-=eta *self.grad_w
        self.b-=eta *self.grad_b
#--输出层
class OutputLayer:
    def __init__(self,n_upper,n):
        self.w = wb_width * np.random.randn(n_upper, n) #权重矩阵
        self.b =wb_width *np.random.randn(n) #偏置向量

    def forward(self,x):
        self.x=x
        u=np.dot(x, self.w)+self.b
        self.y=np.exp(u)/np.sum(np.exp(u),axis=1,keepdims=True)  #softmax函数

    def backward(self,t):
        delta= self.y - t
        self.grad_w= np.dot(self.x.T, delta)
        self.grad_b= np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)

    def update(self,eta):
        self.w-= eta * self.grad_w
        self.b-=eta *self.grad_b
#--各个网络层的初始化一
middle_layer=MiddleLayer(n_in,n_mid)  #MiddleLayer(2,6)
output_layer=OutputLayer(n_mid,n_out) #OutputLayer(6,2)
#-学习—一
sin_data=np.sin(np.pi *X) #对结果验证
#用于对结果的验证
for i in range(epoch):
    #将索引值随机打乱排序
    #反向传播
    index_random =np.arange(n_data)
    np.random.shuffle(index_random) #处理随机数
    #用于结果的显示
    total_error=0
    x_1=[]
    y_1=[]
    x_2=[]
    y_2=[]
    for idx in index_random: #随机索引值，遍历441次
        x = input_data[idx]  #  input_data的第idx行，x=[xi,yi]，x是numpy.ndarray（一维数组）
        t = correct_data[idx]
        # 前向传播
        # reshape 操作前,x是numpy.ndarray（一维数组），形状为 (2,)，即 [x, y]。
        # reshape(1, 2)后,x 变成了numpy.ndarray(二维数组)，形状为 (1, 2)，即 [[x, y]]。
        middle_layer.forward(x.reshape(1, 2))
        output_layer.forward(middle_layer.y)

        # 反向传播
        output_layer.backward(t.reshape(1, 2))
        middle_layer.backward(output_layer.grad_x)

        # 更新参数
        middle_layer.update(eta)
        output_layer.update(eta)
        if i%interval ==0:
            y=output_layer.y.reshape(-1)  #将矩阵还原成向量
            #1e -7   是为了防止log(0)的情况发生 ,分类问题的损失用交叉熵误差损失公式为：l= -∑ t * log(y)    
            total_error +=-np.sum(t *np.log(y+1e-7)) #误差的计算
            #交叉熵误差
            #对概率的大小进行比较并分类
            if y[0]>y[1]:
                x_1.append(x[0])
                y_1.append(x[1])
            else:
                x_2.append(x[0])
                y_2.append(x[1])
    if i%interval==0:
        #显示输出结果的图表
        plt.plot(X,sin_data,linestyle="dashed")
        plt.scatter(x_1,y_1,marker="+")
        plt.scatter(x_2,y_2,marker="x")
        plt.show()
        #显示epoch次数和误差
        print("Epoch:"+ str(i) +"/"+str(epoch),"Error:"+ str(total_error/n_data))# 平均交叉熵
    