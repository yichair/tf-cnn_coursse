'''
分类：与回归不同，获得的结果一般是一个矩阵表示分类，结果由预测值
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #导入TensorFlow自带的数字图像数据

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)# 如果电脑里没有数据集1到10则下载

# 定义训练层
def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 定义模型精确度计算函数
def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict = {xs: v_xs})# 计算test集数据预测值
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))# 对比预测值和真实数据差别
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))# 计算有多少个对的
	result = sess.run(accuracy, feed_dict = {xs: v_xs, ys: v_ys})
	return result # result是一个百分比，百分比越高越准确


# 定义输入数据的placeholder
xs = tf.placeholder(tf.float32, [None, 784]) # 输入数据为28x28像素点的图片，即为784大小的矩阵，None表示不限制输入数据个数
ys = tf.placeholder(tf.float32, [None, 10]) # 输出数据为10行的列向量，哪行为1则表示数字为哪行行数

# 添加输出层
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# 计算分类结果和真实数据的差距并进行优化
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
											  reduction_indices=[1])) # 代价函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

# 初始化所有变量
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# 训练1000步
for i in range(1000):
	barch_xs, barch_ys = mnist.train.next_batch(100)# 从导入的数据集mnist中提取100个数据.mnist数据集分为训练数据和测试数据
	sess.run(train_step, feed_dict = {xs: barch_xs, ys:barch_ys})
	if i % 50 == 0:
		print(compute_accuracy(
			mnist.test.images, mnist.test.labels))# 每50步测试一下我们训练的模型在test数据集上的精确度

