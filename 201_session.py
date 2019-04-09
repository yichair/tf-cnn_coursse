"""
session可以指向神经网络结构的某一部分，例如变量，然后用run激活即执行
"""
import tensorflow as tf

m1 = tf.constant([[2, 2]]) 	#行向量
m2 = tf.constant([[3],		#列向量
                  [3]])
dot_operation = tf.matmul(m1, m2)#矩阵x1，x2相乘==np.dot(m1,m2)

# method1 use session
sess = tf.Session()
result = sess.run(dot_operation)
print(result)
sess.close()	# 可以没有

# method2 use session
with tf.Session() as sess:		#运用with语句，运行后自动关闭sess
    result_ = sess.run(dot_operation)
    print(result_)