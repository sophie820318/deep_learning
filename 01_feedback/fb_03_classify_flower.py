import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#--鸢尾花数据的读入一-鸢尾花数据集共有 150 个样本，3 个类别，每类 50 个样本。
iris_data = datasets.load_iris()
input_data =iris_data.data
correct=iris_data.target
n_data =len(correct) #150个
#样本数量
#--对输入数据进行标准化处理
ave_input=np.average(input_data,axis=0) #numpy.ndarray(n,) 表示“按列”进行操作 [均值1,均值2,均值3,均值4] 每个特征的标准差
std_input=np.std(input_data,axis=0) 
input_data =(input_data - ave_input) / std_input
#--将正确答案转换为独热编码格式-
correct_data =np.zeros((n_data,3))
for i in range(n_data):
    correct_data[i, correct[i]] =1.0
    # 独热编码
    #[1,0,0]
    #[0,1,0]
    #[0,0,1]
#--训练数据与测试数据--
#numpy.ndarray(一维数组)内容为从0到n_data-1的整数序列
index=np.arange(n_data) 
#选取偶数下标作为训练数据 
index_train =index[index%2==0]
#选取奇数下标作为测试数据
index_test=index[index%2!=0]
#训练输入
input_train = input_data[index_train,:]
#训练正确答案
correct_train = correct_data[index_train,:]
#测试输入
input_test= input_data[index_test,:]
correct_test = correct_data[index_test, :]

#page2
n_train =input_train.shape[0]  #shape[0] 表示数组的第0维长度，也就是行数，即训练样本的数量。150/2=75
#训练数据的样本数
n_test = input_test.shape[0]
#测试数据的样本数
#--各种设置值
#输入层的神经元数量，花的数据，花瓣2个特征，花萼2个特征，共4个特征
n_in=4
#中间层的神经元数量
n_mid=25
#输出层的神经元数量
n_out=3
#各对鸢属
wb_width=0.1
#权重和偏置的分散度
eta=0.01
#学习系数
epoch=50
batch_size=8 
interval=100
#显示进度的时间间隔
#--各个网络层的祖先类一-
class BaseLayer:
    def __init__(self,n_upper,n):
        self.w= wb_width * np.random.randn(n_upper, n) #权重
        self.b=wb_width *np.random.randn(n)
#偏置
    def update(self,eta):
        self.w-=eta *self.grad_w
        self.b-= eta *self.grad_b
#--中间层
class MiddleLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        self.u = np.dot(x, self.w) + self.b
        #ReLU
        self.y = np.where(self.u <= 0, 0, self.u)
    def backward(self,grad_y):
        #ReLU的微分
        delta =grad_y *np.where(self.u<=0, 0, 1)
        self.grad_w = np.dot(self.x.T,delta)
        self.grad_b=np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)
#--输出层
class OutputLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        self.u = np.dot(x, self.w) + self.b
        #Softmax
        exp_u = np.exp(self.u - np.max(self.u, axis=1, keepdims=True))
        self.y = exp_u / np.sum(exp_u, axis=1, keepdims=True)#
#page3
    def backward(self,t):
        delta =self.y-t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b=np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)

#--各个网络层的初始化一
middle_layer_1=MiddleLayer(n_in,n_mid)
middle_layer_2=MiddleLayer(n_mid,n_mid)
output_layer= OutputLayer(n_mid,n_out)
#--正向传播-一
def forward_propagation(x):
    middle_layer_1.forward(x)
    middle_layer_2.forward(middle_layer_1.y)
    output_layer.forward(middle_layer_2.y)
    
#--逆向传播
def backpropagation(t):
    output_layer.backward(t)
    middle_layer_2.backward(output_layer.grad_x)
    middle_layer_1.backward(middle_layer_2.grad_x)
#--权重和偏置的更新--
def update_wb():
    middle_layer_1.update(eta)
    middle_layer_2.update(eta)
    output_layer.update(eta)
#--计算交叉熵误差一一
def get_error(t,batch_size):
    return -np.sum(t * np.log(output_layer.y + 1e-7))/batch_size
#--用于记录误差一
train_error_x=[]
train_error_y=[]
test_error_x=[]
test_error_y=[]
#-记录学习的过程和经过一
n_batch =n_train // batch_size
#每一轮epoch的批次尺寸
for i in range(epoch):
    #误差的统计
    forward_propagation(input_train)
    error_train = get_error(correct_train, n_train)
    forward_propagation(input_test)
    error_test = get_error(correct_test, n_test)
    #误差的记录
    test_error_x.append(i)
    test_error_y.append(error_test)
    train_error_x.append(i)
    train_error_y.append(error_train)
    #-—进度的显示一
    if i%interval ==0:
        print("Epoch:"+ str(i) +"/"+ str(epoch),
        "Error_train:"+str(error_train),
        "Error_test:"+str(error_test))
    #学习—一
    index_random=np.arange(n_train)
    #将索引值随机打乱排序
    np.random.shuffle(index_random)
    for j in range(n_batch):
        #取出最小批次
        mb_index=index_random[j*batch_size:(j+1)*batch_size]
        x=input_train[mb_index,:]
        t=correct_train[mb_index,:]
        #正向传播和反向传播
        forward_propagation(x)
        backpropagation(t)
        #权重和偏置的更新
        update_wb()
    #--使用图表显示误差记录-
    plt.plot(train_error_x, train_error_y, label="Train")
    plt.plot(test_error_x, test_error_y, label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

    #page5
    forward_propagation(input_train)
    count_train = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_train, axis=1))
    forward_propagation(input_test)
    count_test = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_test, axis=1))
    print("Accuracy Train:", str(count_train/n_train*100) + "%")
    print("Accuracy Test:", str(count_test/n_test*100) + "%")