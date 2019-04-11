'''
过拟合则泛化能力下降，为了解决过拟合，方法如下：
1、增加训练数据
2、正则化:
	l1: cost=(Wx - realy)^2 + abs(W) #abs(W)表示W的绝对值
	l2: cost=(Wx - realy)^2 + (W)^2
	...
	神经网络正则化Dropout regularization:用不完整的神经网络训练一次,第二次再忽略一些
'''
import tensorflow as tf
 # 从sklearn数据库导入0到9的图片数据,没有需要先安装
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y) # 把y变成二进制
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)# 把X y分为训练集和测试集

# 定义层
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs)# 一定要有？
    return outputs


# 定义神经网络输入数据的placeholder
keep_prob = tf.placeholder(tf.float32) # keep_prob是保留概率，即要保留的结果所占的比例
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# 设置隐藏层和输出层
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)# 输入图像为8*8
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，交叉熵就等于零
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy) # 记录交叉熵变化情况
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

# 显示traindata 和 testdata的变化曲线
train_writer = tf.summary.FileWriter("logs/overfitting/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/overfitting/test", sess.graph)

# 初始化变量
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    # 开始训练
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})# 当keep_prob=1时，模型对训练数据的适应性优于测试数据，dropout没起作用
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)