import tensorflow as tf
import numpy as np
X_data = np.random.rand(100).astype(np.float32)
y_data = X_data*0.1 + 0.3

Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weight*X_data + biases
# 误差值,代价函数，真实值和预测值相减后平方的平均数；
loss = tf.reduce_mean(tf.square(y-y_data))

# 优化器，用来减小误差（参数为学习效率）,梯度下降法；
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 最小化代价函数，loss.
train = optimizer.minimize(loss)

# 初始化所有变量；
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 ==0:
            print(step, sess.run(Weight), sess.run(biases))
