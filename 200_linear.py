import tensorflow as tf
import numpy as np # 数学计算模块，存储和处理大型矩阵

# 创建数据
x_data = np.random.rand(100).astype(np.float32)# 生成一百个随机数列，TensorFlow数据一般都是float32
y_data = x_data*0.1 + 0.3# 结果

### 开始创建TensorFlow结构 ###
#通过学习接近0.1 0.3
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))#定义参数变量，大写开头可能是矩阵，通过随机数列生成，一维的，范围为-1到1
biases = tf.Variable(tf.zeros([1]))#初始值为0
y = Weights*x_data + biases # 预测函数,y为预测值

loss = tf.reduce_mean(tf.square(y-y_data))#计算预测值和真实值差别
optimizer = tf.train.GradientDescentOptimizer(0.5)#选择通过阶梯下降方式进行优化的优化器，0.5为学习效率
train = optimizer.minimize(loss)#优化器减小误差

init = tf.global_variables_initializer()#初始化以上定义的变量
### 结束创建TensorFlow结构 ###

#激活神经网络结构
sess = tf.Session()
sess.run(init)#session相当于一个指针，指向init变量，激活神经网络结构

for step in range(201):# 训练201步
	sess.run(train)
	if step % 20 == 0:
		print(step,sess.run(Weights),sess.run(biases))#每20步输出Weights、biases