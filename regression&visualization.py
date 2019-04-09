"""
添加或定义一个新的神经层
主要定义：权重，激励函数

建立神经网络结构，分别定义每层，计算和真实差异然后进行提升

可视化训练结果

用with进行结构可视化，用 Tensorflow 自带的 tensorboard 画出我们所建造出来的神经网络的流程图

用histogram和scatter可视化训练过程中变量的变化（histogram还有点问题）
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

# 定义训练层
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None): #没有激励函数则默认激励函数是一个线性函数
	layer_name = 'layer%s' % n_layer
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size])) # 这个权重矩阵为随机变量比较好
			# tf.histogram_summary(layer_name+'/weights',Weights)
			tf.summary.histogram(layer_name + '/weights', Weights)# 定义关于第几层weights的直方图,观看它的变化，颜色深的地方表示数据多，以下同理
		with tf.name_scope('biases'):	
			biases = tf.Variable(tf.zeros([1,out_size])+ 0.1,name='b')#列表,biases初始值不为零所以
			tf.summary.histogram(layer_name + '/biases', biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs, Weights) + biases
		# 如果激励函数为线性函数，权重和输入数据直接相乘,否则要使用激励函数
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		tf.summary.histogram(layer_name + '/outputs', outputs)
	return outputs

# 设置训练数据
x_data = np.linspace(-1,1,300)[:,np.newaxis] #用linspace()产生随机数字，-1到1区间有300个单位，表示300行即列，即300个样本1个特征
noise = np.random.normal(0,0.05,x_data.shape) #为了和真实数据一样，进行加噪，y的平均数为0，方差为0.05，格式和x_data一样
y_data = np.square(x_data) - 0.5 + noise #假设y_data为x_data的2次方-0.5

# 定义输入数据
with tf.name_scope('inputs'):#整个输入的统称为inputs
	xs = tf.placeholder(tf.float32,[None,1],name='x_input') # None代表无论输入多少个样本都可以，1代表只有一个特征
	ys = tf.placeholder(tf.float32,[None,1],name='y_input')

# 建立输入层（多少个特征多少神经元），隐藏层(神经元个数自己假设)，输出层（多少个特征多少神经元)
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu) #输入层到隐藏层
predition = add_layer(l1, 10, 1 , n_layer=2, activation_function=None) #隐藏层到输出层

# 设置损失函数和训练
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
			reduction_indices=[1])) # 每个样本真实值-预测值的平方求和再求平均数
	tf.summary.scalar('loss', loss)# loss是一个标量的变化，所以不在histogram中显示，而在event中
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 学习效率假设为0.1，通常小于1,表示训练的每一步都是通过前者的优化器，以0.1的效率对误差进行最小化，再下一次得到更好的结果

sess = tf.Session()
#合并所有定义的summary
merged = tf.summary.merge_all()
# 可视化网络结构：tf.train.SummaryWriter这种方式在TensorFlow0.12版本后被替用，所以我们需要判断版本
if int((tf.__version__).split('.')[0]) < 1 and int((tf.__version__).split('.')[1]) < 12:  #版本<0.12
	writer = tf.train.SummaryWriter("./logs/",sess.graph)# 把整个框架加载到文件里，然后才能从文件里加载在浏览器中观看,graph是指整个框架
else:
	writer = tf.summary.FileWriter("./logs/",sess.graph)
# 运行之后我们在当前目录下面运行命令行 tensorboard --logdir='logs/'，然后打开网址http://SC-201708311248:6006即可看见我们的框架视图
# 初始化所有的变量
init = tf.global_variables_initializer()
sess.run(init)

# 显示数据图形
fig = plt.figure()# 生成图片框
ax = fig.add_subplot(1,1,1)# 连续性画图1,1,1为它的编号
ax.scatter(x_data,y_data)#使真实数据在图上进行分布
plt.ion()# show后不暂停
plt.show()# show()函数画好后将整个函数暂停

#训练1000步，每次激活训练函数
for i in range(1000):
	sess.run(train_step, feed_dict={xs: x_data, ys: y_data})#有些训练xs不是全部的x_data
	if i % 50 == 0:
		# print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
		# 尝试抹除前一条，第一次没有则报错
		try:
			ax.lines.remove(lines[0])#移除lines第一条线
		except Exception:
			pass
		# 绘制结果图
		prediction_value = sess.run(predition, feed_dict = {xs: x_data})
		lines = ax.plot(x_data,prediction_value,'r-',lw=5)#连续画线，宽度为5
		plt.pause(0.1)#暂停0.1秒

		# 显示每次检测的变量图形
		summary_result = sess.run(merged, feed_dict = {xs: x_data, ys: y_data})
		writer.add_summary(summary_result, i) # i表示步数，在x轴上显示
