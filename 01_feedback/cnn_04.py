import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#--手写数字数据集的读入一,8x8像素的手写数字图像数据集
digits_data = datasets.load_digits() #(1797,64)的numpy.ndarray数组)
input_data = digits_data.data #(1797,64)的numpy.ndarray
correct = digits_data.target  #也就是一维数组，长度为 1797，每个元素是一个样本的标签（0~9 的整数）。
n_data = len(correct)

#page2
#——输入数据的标准化
#所有元素（即所有像素值）加起来，然后除以总元素个数（1797 × 64），得到一个标量

ave_input=np.average(input_data)  
std_input=np.std(input_data)
#这行代码会让所有像素值减去均值，再除以标准差，使得处理后的数据均值为0，标准差为1。
input_data =(input_data-ave_input) / std_input
# 归一化常规处理手段
#--将正确答案转换为独热编码格式的处理手段
correct_data =np.zeros((n_data,10))
for i in range(n_data):
    correct_data[i,correct[i]] =1.0
#--训练数据与测试数据--
index=np.arange(n_data)#一维的NumPy数组.内容是从0到n_data-1的整数序列
index_train=index[index%3 !=0 ] #训练数据占三分之二,1797*2/3=1198
index_test=index[index%3 ==0 ]  #测试数据占三分之一1797/3 =599
input_train =input_data[index_train,:]
#训练输入数据
correct_train = correct_data[index_train,:]
#训练正确答案
input_test= input_data[index_test,:]
#测试输入数据
correct_test= correct_data[index_test,:]
#测试 正确答案
n_train=input_train.shape[0]
#训练数据的采样数
#测试数据的采样数
n_test=input_test.shape[0]

#接下来，对学习相关的各个值进行设置。
#--各个设置值
#输入图像的高度
img_h=8
#输入图像的宽度
img_w=8
#输入图像的通道数
img_ch=1
#权重与偏置的扩散度
wb_width=0.1
#学习系数
eta=0.01
epoch=50
batch_size=8
#显示进度的间隔时间
interval=10
#误差计算的采样数
n_sample=200
#然后,对im2col 和col2im函数定义。关于这两个函数的代码、在之前的小群
#page3-4
#--im2col 将图像转换为矩阵
def im2col(images,flt_h,flt_w,out_h,out_w,stride,pad):
    n_bt,n_ch,img_h,img_w=images.shape
    # 第1维（batch）和第2维（通道）不填充，第3维（高）和第4维（宽）在两侧各填充 pad 个像素，填充值为0（"constant"）
    img_pad=np.pad(images, [(0,0),(0,0),(pad, pad),(pad, pad)],"constant")
    cols = np.zeros((n_bt, n_ch,flt_h,flt_w,out_h, out_w))

    for h in range(flt_h):
        h_lim=h+stride*out_h
        for w in range(flt_w):
            w_lim=w+stride*out_w
            cols[:,:,h,w,:,:]=img_pad[:, :, h:h_lim:stride, w:w_lim:stride]
    cols=cols.transpose(1, 2, 3, 0, 4, 5).reshape(n_ch*flt_h*flt_w, n_bt*out_h*out_w)
    return cols
#----col2im-----
def col2im(cols,img_shape,flt_h,flt_w,out_h,out_w,stride,pad):
    n_bt,n_ch,img_h,img_w= img_shape
    cols=cols.reshape(n_ch,flt_h,flt_w,n_bt, out_h, out_w).transpose(3,0,1,2,4,5)
    images=np.zeros((n_bt,n_ch,img_h+2*pad+stride-1,img_w+2*pad+stride-1))
    for h in range(flt_h):
        h_lim=h+stride*out_h
        for w in range(flt_w):
            w_lim =w+ stride*out_w
            images[:, :, h:h_lim:stride, w:w_lim:stride]+=cols[:, :,h,w,:,:]
    return images[:,:,pad:img_h+pad,pad:img_w+pad]

#page5-6
# 卷积层和池化层是作为类进行封装实现的
class ConvLayer:
    def __init__(self,x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad):
        #将参数集中保存
        self.params=(x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad)
        #过滤器和偏置的初始值
        self.w=wb_width *np.random.randn(n_flt,x_ch,flt_h,flt_w) #NumPy 的多维数组，装数字的容器
        self.b = wb_width * np.random.randn(1, n_flt)
        #卷积层输出的通道数，等于滤波器的数量
        self.y_ch=n_flt
        #卷积层输出的高度的标准公式
        # 1.x_h - flt_h：输入高度减去滤波器高度，得到“可移动的距离”
        # 2.+ 2*pad：加上两边的填充（上下各 pad 个像素）
        # 3.// stride：除以步长，得到“可以放置多少个滤波器位置”
        # 4.+ 1：加1是因为包含起始位置
        self.y_h=(x_h-flt_h+2*pad)//stride+1
        #输出的宽度
        self.y_w=(x_w-flt_w+ 2*pad) // stride+1
        #AdaGrad算法用，根据历史梯度自适应，h_w = 权重 w 的“历史梯度平方累积”
        self.h_w=np.zeros((n_flt,x_ch,flt_h,flt_w))+1e-8
        #AdaGrad算法用，根据历史梯度自适应，h_b = 偏置 b 的“历史梯度平方累积”
        self.h_b=np.zeros((1,n_flt))+ 1e-8
    def forward(self,x):
        #获取批次大小
        n_bt=x.shape[0]
        x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad=self.params
        #获取卷积层输出的通道数、高度和宽度
        y_ch,y_h,y_w=self.y_ch,self.y_h,self.y_w
        #将输入图像和过滤器转换成矩阵
        self.cols=im2col(x,flt_h,flt_w,y_h,y_w,stride,pad)
        self.w_col=self.w.reshape(n_flt,x_ch*flt_h*flt_w)
        #输出的计算：矩阵乘积、偏置的加法运算、激励函数
        #(n_flt,x_ch*flt_h*flt_w) * (x_ch*flt_h*flt_w, n_bt*y_h*y_w) = (n_bt,y_h, y_w, y_ch)
        #【(n_flt, n_bt*y_h*y_w).T】.T = (n_bt*y_h*y_w, n_flt)
        u=np.dot(self.w_col, self.cols).T + self.b
        #将一维化的数据恢复为4维张量：
        #原始：(n_bt*y_h*y_w, n_flt) — 所有空间位置和批次被展平
        #reshape 后：(n_bt, y_h, y_w, y_ch) — 恢复为批次、高度、宽度、通道的4维结构
        #(n_bt,y_ch,y_h, y_w)
        self.u =u.reshape(n_bt,y_h, y_w, y_ch).transpose(0,3,1,2)
        #RELU激活函数
        self.y =np.where(self.u <= 0, 0, self.u) 
    def backward(self,grad_y):
        #获取批次大小
        n_bt=grad_y.shape[0]
        x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad=self.params
        y_ch,y_h,y_w= self.y_ch,self.y_h,self.y_w
        #delta= (n_bt,y_ch,y_h, y_w)
        delta = grad_y*np.where(self.u <=0,0,1)  
        #(n_bt,y_h, y_w,y_ch),transpose 把最后一维拆出来，reshape 展平，得到(n_bt*y_h*y_w,y_ch)
        delta = delta.transpose(0,2,3,1).reshape(n_bt*y_h*y_w,y_ch)
        #过滤器和偏置的梯度
        grad_w= np.dot(self.cols,delta)
        self.grad_w= grad_w.T.reshape(n_flt,x_ch,flt_h,flt_w)
        #沿着第0维相加，把这一维“抹掉”,对每一列求和
        self.grad_b=np.sum(delta,axis=0)
        #输入的梯度
        grad_cols = np.dot(delta, self.w_col)
        x_shape=(n_bt,x_ch,x_h,x_w)
        self.grad_x= col2im(grad_cols.T,x_shape,flt_h,flt_w,y_h,y_w, stride,pad)
    def update(self,eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b

#--池化层-——
class PoolingLayer:
    #nbt:批次尺寸,xch：输入的通道数量，xh:输入图像的高度，x_w:输入图
    #poo:池化区域的尺寸，pad:填充的幅度
    #y.ch:输出的通道数量, y_h:输出的高度, y_w:输出的宽度
    def __init__(self, x_ch, x_h,x_w, pool, pad):
        #将参数集中保存
        self.params =(x_ch,x_h,x_w,pool, pad)
        #输出图像的尺寸
        #输出的通道数
        self.y_ch=x_ch
        #输出的高度
        self.y_h=x_h//pool if x_h%pool==0 else x_h//pool+1
        self.y_w=x_w//pool if x_w%pool==0 else x_w//pool+1 #输出的宽度
    def forward(self, x):
        n_bt =x.shape[0]
        x_ch,x_h,x_w,pool, pad = self.params
        y_ch,y_h, y_w = self.y_ch, self.y_h,self.y_w
        #将输入图像转换成矩阵
        cols= im2col(x, pool, pool, y_h, y_w, pool, pad)
        cols = cols.T.reshape(n_bt*y_h*y_w*x_ch, pool*pool)
        #输出的计算：最大池化
        y=np.max(cols,axis=1)
        self.y = y.reshape(n_bt, y_h, y_w, x_ch).transpose(0, 3, 1, 2)
        #保存最大值的索引值
        self.max_index = np.argmax(cols, axis=1)
    def backward(self,grad_y):
        n_bt=grad_y.shape[0]
        x_ch,x_h,x_w,pool,pad =self.params
        y_ch,y_h,y_w=self.y_ch,self.y_h,self.y_w
        #对输出的梯度的坐标轴进行切换
        grad_y=grad_y.transpose(0,2,3,1)
        #创建新的矩阵，只对每个列中具有最大值的元素所处位置中放入输出的
        grad_cols =np.zeros((pool*pool, grad_y.size))
        grad_cols[self.max_index.reshape(-1), np.arange(grad_y.size)] =grad_y.reshape(-1)
        grad_cols=grad_cols.reshape(pool,pool, n_bt, y_h,y_w,y_ch)

        grad_cols =grad_cols.transpose(5,0,1,2,3,4)
        grad_cols=grad_cols.reshape( y_ch*pool*pool, n_bt*y_h*y_w)
        #输入的梯度
        x_shape=(n_bt,x_ch,x_h,x_w)
        self.grad_x= col2im(grad_cols,x_shape,pool,pool,y_h,y_w,pool,pad)
#全连接层也是作为类来进行封装和实现的。
#--全连接层的祖先类-一
class BaseLayer:
    def __init__(self,n_upper,n):
        self.w = wb_width *np.random.randn(n_upper, n)
        self.b=wb_width *np.random.randn(n)
        self.h_w=np.zeros((n_upper,n)) + 1e-8
        self.h_b=np.zeros(n)+1e-8
    def update(self,eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        self.h_b += self.grad_b *self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b
    #--全连接的中间层一-
class MiddleLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0,0, self.u)
    def backward(self,grad_y):
        delta = grad_y *np.where(self.u <= 0,0,1)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b =np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)
    #一-全连接的输出层
class OutputLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        u =np.dot(x, self.w) + self.b
        self.y=np.exp(u) / np.sum(np.exp(u), axis=1).reshape(-1,1)
    def backward(self, t):
        delta = self.y-t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
#page 13       
#接下来，构建CNN 网络。
#--各个网络层的初始化一
cl_1=ConvLayer(img_ch,img_h,img_w,10,3,3,1,1)
pl_1=PoolingLayer(cl_1.y_ch,cl_1.y_h,cl_1.y_w,2,0)
n_fc_in=pl_1.y_ch *pl_1.y_h*pl_1.y_w
ml_1=MiddleLayer(n_fc_in,100)
ol_1=OutputLayer(100,10)

#--正向传播-—
def forward_propagation(x):
    n_bt=x.shape[0]
    images =x.reshape(n_bt, img_ch,img_h, img_w)
    cl_1.forward(images)
    pl_1.forward(cl_1.y)
    fc_input=pl_1.y.reshape(n_bt,-1)
    ml_1.forward(fc_input)
    ol_1.forward(ml_1.y)
    #一—反向传播
def backpropagation(t):
    n_bt=t.shape[0]
    ol_1.backward(t)
    ml_1.backward(ol_1.grad_x)
    grad_img=ml_1.grad_x.reshape(n_bt,pl_1.y_ch,pl_1.y_h,pl_1.y_w)
    pl_1.backward(grad_img)
    cl_1.backward(pl_1.grad_x)
#-权重和偏置的更新一
def update_wb():
    cl_1.update(eta)
    ml_1.update(eta)
    ol_1.update(eta)
    #一－对误差进行计算
def get_error(t,batch_size):
    #交叉熵误差
    return -np.sum(t*np.log(ol_1.y+1e-7)) /batch_size
    #--对样本进行正向传播
def forward_sample(inp,correct,n_sample):
    index_rand =np.arange(len(correct))
    np.random.shuffle(index_rand)
    index_rand=index_rand[:n_sample]
    x=inp[index_rand,:]
    t=correct[index_rand,:]
    forward_propagation(x)
    return x,t
#接下来，使用构建完毕的 CNN 网络进行学习。每完成一轮epoch，将对训练误差和测试误差进行计算并将其记录下来
#用于对误差进行记录
train_error_x=[]
train_error_y=[]
test_error_x=[]
test_error_y=[]
#--用于对学习过程进行记录-，
#商取整
n_batch=n_train //batch_size
for i in range(epoch):
    #--误差的测算
    x,t=forward_sample(input_train,correct_train,n_sample)
    error_train =get_error(t,n_sample)
    x,t=forward_sample(input_test,correct_test,n_sample)
    error_test=get_error(t,n_sample)
    #--误差的记录一
    train_error_x.append(i)
    train_error_y.append(error_train)
    #page 18
    test_error_x.append(i)
    test_error_y.append(error_test)
#-处理进度的显示-—
if i%interval==0:
    print("epoch:" + str(i) + "/" + str(epoch),
    "Error_train:" + str(error_train),
    "Error_test:" + str(error_test))
#学习交叉熵误差

index_rand=np.arange(n_train)
np.random.shuffle(index_rand)
for j in range(n_batch):
    mb_index=index_rand[j*batch_size:(j+1)*batch_size]
    x=input_train[mb_index,:]
    t=correct_train[mb_index,:]
    forward_propagation(x)
    backpropagation(t)
    update_wb()
#最后, 对学习的结果进行显示。

#一显示记录误美的表格

plt.plot(train_error_x,train_error_y,label="Train")
plt.plot(test_error_x,test_error_y,label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

x,t=forward_sample(input_train,correct_train,n_train)
count_train =np.sum(np.argmax(ol_1.y,axis=1)== np.argmax(t,axis=1))

x,t=forward_sample(input_test,correct_test,n_test)
count_test =np.sum(np.argmax(ol_1.y,axis=1)== np.argmax(t,axis=1))
print("Accuracy Train:",str(count_train/n_train*100)+"%",
"Accuracy Test:",str(count_test/n_test*100)+"%")