"""
现实生活中很多模型并不是线性的：y=Wx,所以我们需要激励函数进行改造:y=AF(Wx),其中AF()就是激励函数
激励函数AF()一般来说可能是relu(),sigmoid(),tanh()
卷积神经网络推荐：relu()
循环神经网络推荐：relu(),tanh()
激励函数需要可微分
激励函数作用让某一部分神经元激活，将激活信息传递到后面的神经元上
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #绘图库

# fake data
x = np.linspace(-5, 5, 200)     # x data, shape=(100, 1)

# following are popular activation functions
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)
# y_softmax = tf.nn.softmax(x)  softmax is a special kind of activation function, it is about probability

sess = tf.Session()
y_relu, y_sigmoid, y_tanh, y_softplus = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus])

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))# 绘图的图框大小

plt.subplot(221)	#子图在左上角
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))# y轴范围-1到5
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()