import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples//batch_size

# 初始化权值；
def weight_varible(shape):
    # 生成一个截断正态分布（高斯分布）；
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化偏置值
def bias_varible(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层：
def conv2d(x, w):
    # x:输入采样图片的张量形式：[图片批次,图片长,图片宽,图片通道数]
    # w:滤波器（卷积核），4维张量形式[卷积核长，宽，输入的通道数，输出的通道数]
    # strides:步长，4维向量：0和4为固定值1，中间两个位置为长宽；
    # padding:采样方式：SAME,VALID;
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# 池化层：
def max_pool_2X2(x):
    # ksize; 池化的大小为2*2，0位和3位为固定值1；
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义两个占位符用来存储训练样本和标签；
x = tf.placeholder(tf.float32, [None, 784])
# 因为图片为28*28的图片，0位置为图片的张数（样本数量）；
y = tf.placeholder(tf.float32, [None, 10])  # 训练集的标签是10个（0~9数字）；

# 把采样数据x转换成4维张量，：[批次，长，宽，通道]
x_iamge = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层的权重值和偏置值：
W_conv1 = weight_varible([5, 5, 1, 32])  # 5*5的采样窗口，通道数为1（黑白图），32个卷积核从一个平面抽取特征；
# 给每一个卷积核配一个偏置值；
b_conv1 = bias_varible([32])

# 把x_image放入到卷积层中进行卷积处理；
h_conv1 = tf.nn.relu(conv2d(x_iamge, W_conv1) + b_conv1)
# 把得到的结果进行池化；
h_pool1 = max_pool_2X2(h_conv1)

# 进行第二次卷积，池化；
# 初始化第二个卷积层的权值与偏置值：
W_conv2 = weight_varible([5, 5, 32, 64])  # 同样5*5的窗口，因为上一次卷积有32个卷积层，所以这一次定义64个卷积核在32个平面取样；
b_conv2 = bias_varible([64])  # 每个卷积核配一个偏置值；
# 导入上一次卷积池化后的结果，然后加上新的权重值和偏置值；
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 再次池化；
h_pool2 = max_pool_2X2(h_conv2)

# 第一次卷积后[28,28],第一次池化：[14,14],第二次卷积：[14,14],第二次池化：[7*7]，并且第二次卷积后有64个平面；
# 进行全连接层的初始化：
# 全连接层的权重值：第二次池化后的结果转化成一个一维张量，并且链接定义好的1024个全连接神经元；
W_fc1 = weight_varible([7*7*64, 1024])
# 每个神经元分配一个偏置值；
b_fc1 = bias_varible([1024])

# 把第二次池化后的形状转换成一维扁平化的形状；
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 把这个结果带入到全连接层中用relu激活函数输出；
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 进行dropout操作：
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 进行第二次全连接层的初始化；
# 第二次全连接层权值定义，接上一次输出的神经元个数1024，最终标签数量为10个；
W_fc2 = weight_varible([1024, 10])
b_fc2 = bias_varible([10])   # 每个神经元分配一个偏置值；

# 计算输出；
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 代价函数的计算，使用交叉熵；
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# loss = -tf.reduce_sum(y_data*tf.log(y_model))
# 使用优化器进行优化；Adam函数,学习率为10 的-4次方
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求得准确率；
# 结果存放在一个布尔列表中：
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化一个会话：
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 得到数据集特征和标签列表；
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})  # 每次训练有70%的神经元参与；

        # 训练完成一个周期后，进行一次测试，并且输出测试值；
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        T = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(str(T)+"训练周期：" + str(epoch) + "  测试得分：" + str(acc))













