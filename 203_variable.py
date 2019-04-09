"""
TensorFlow定义变量:
tf.Variable(初始值，name='某某')
"""
import tensorflow as tf

var = tf.Variable(0)    # our first variable in the "global_variable" set
one = tf.constant(1)
add_operation = tf.add(var, one)# 变量加常量还是变量
update_operation = tf.assign(var, add_operation)#将add_operation赋值到var上

init = tf.global_variables_initializer()#初始化变量

with tf.Session() as sess:
    # once define variables, you have to initialize them by doing this
    sess.run(init)
    for _ in range(3):
        sess.run(update_operation)
        print(sess.run(var))