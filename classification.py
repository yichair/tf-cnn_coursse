import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #导入TensorFlow自带的数字图像数据

mnist = input_data.read_data_sets('MNIST_data', one_hot=Ture)# 如果电脑里没有数据集1到10则下载

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