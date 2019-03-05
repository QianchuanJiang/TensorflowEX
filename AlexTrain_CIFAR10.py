# coding: utf-8
import numpy as np
import tensorflow as tf
import time
from CIFARreader import RaadTrianData
from CIFARreader import unpickle


display_step = 50
# 读取并载入训练数据集的方法；
X_train, y_train = RaadTrianData()
batch_size = 1000


# 编写一个整数数字转换成OneHot的方法；
def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels
# print(onehot([0,5,8,6,8,10]))
# 开始导入训练集；
'''
data1 = unpickle('cifar10-dataset/data_batch_1')
data2 = unpickle('cifar10-dataset/data_batch_2')
data3 = unpickle('cifar10-dataset/data_batch_3')
data4 = unpickle('cifar10-dataset/data_batch_4')
data5 = unpickle('cifar10-dataset/data_batch_5')
#  把训练集整合到一个数组中；
X_train = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'], data5['data']), axis=0)
# 把训练集的标签也整合到一个数组中；
y_train = np.concatenate((data1['labels'], data2['labels'], data3['labels'], data4['labels'], data5['labels']), axis=0)
# 标签转换成0neHot形式；
y_train = onehot(y_train)
# 读取测试集特征数据：
test = unpickle('cifar10-dataset/test_batch')
# 只取5000个测试集中的样本用来测验；
X_test = test['data'][:5000, :]
y_test = onehot(test['labels'])[:5000, :]
print(X_test.shape)

# 创建神经网络模型；构建模型参数；
learning_rate = 0.003
training_iters = 200
batch_size = 1000
display_step = 100
# 单个样本大小，图片为32x32像素，彩色3通道的图片，所以32x32x3=3072
n_features = 3072
# 标签种数；
n_classes = 10
# 第一全连接层数量；
n_fc1 = 384
# 第二全连接层数量；
n_fc2 = 192

'''
X_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='input')
y_input = tf.placeholder(dtype=tf.float32, shape=[None], name='label')


def batch_normal(xs, out_size):
    axis = list(range(len(xs.get_shape()) - 1))
    n_mean, n_var = tf.nn.moments(xs, axes=axis)
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    def mean_var_with_update():
        ema_apply_op = ema.apply([n_mean, n_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(n_mean), tf.identity(n_var)

    mean, var = mean_var_with_update()

    bn = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)
    return bn


def AlexNet(image):
    # 定义卷积层1，卷积核大小64*64，偏置量等各项参数参考下面的程序代码，下同。
    with tf.name_scope("conv1") as scope:
        # 权重值，3*3,3通道，24个神经元
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 24], dtype=tf.float32, stddev=0.01, name="weights"))
        # 开始卷积，步长是4x4,pading方式为‘SAME’
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding="SAME")
        # 偏置值分配64个；
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[24]), trainable=True, name="biases")
        # 定义激活函数，用relu激活函数
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
    pass
    # LRN层
    # lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn1")
    # 最大池化层，池化器大小2x2,步长为1x1,pading方式为‘VALID’
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="pool1")
    # 定义卷积层2
    with tf.name_scope("conv2") as scope:
        # 卷积核大小为5x5,卷积神经元数量为192个；
        kernel = tf.Variable(tf.truncated_normal([3, 3, 24, 96], dtype=tf.float32, stddev=0.01, name="weights"))
        # 开始进行卷积，步长为1x1,padding方式为‘SAME’
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
        # 定义偏置值，与卷积核数量相同为192个；
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96]), trainable=True, name="biases")
        # 用relu进行激活函数；
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        pass
    # LRN层
    # lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn2")
    # 最大池化层，池化器大小3x3，步长为2,2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="pool2")

    # 定义卷积层3
    with tf.name_scope("conv3") as scope:
        # 定义卷积核大小为3X3,384个神经元
        kernel = tf.Variable(tf.truncated_normal([3, 3, 96, 192], dtype=tf.float32, stddev=0.01, name="weights"))
        # 进行卷积操作，卷积步长为1*1
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
        # 偏置值同样分配384个；
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[192]), trainable=True, name="biases")
        # 用relu函数激活后得到卷积结果，
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        pass
        # 定义卷积层4，不池化，直接卷积；
    with tf.name_scope("conv4") as scope:
        # 卷积层数量为256个；
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32, stddev=0.01, name="weights"))
        # 卷积操作，步长为1,1；
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
        # 偏置值为256个；
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[192]), trainable=True, name="biases")
        # 使用relu函数进行激活
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        pass

        # 定义卷积层5，与第4层用相同的方式再进行一次卷积操作；
    with tf.name_scope("conv5") as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 96], dtype=tf.float32, stddev=0.01, name="weights"))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        pass

    # 最大池化层，池化器尺寸3,3，步长为2,2，方式为‘VALD’
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="pool5")
    # 进入全连接层，
    # 把第五次池化的结果进行扁平化操作，
    shape = pool5.get_shape()
    print(shape)
    flatten = tf.reshape(pool5, [-1, shape[1].value*shape[2].value*shape[3].value])
    # 设置权重输出为4096个神经元；
    weight1 = tf.Variable(tf.truncated_normal([24 * 24 * 96, 1024], mean=0, stddev=0.01))
    # 定义全连接层的偏置值；
    biasesf1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1024]), trainable=True, name="biases")
    # 进行第一层全连接
    fc1 = tf.matmul(flatten, weight1)
    fcb1 = tf.nn.bias_add(fc1, biasesf1)
    # nfc1 = batch_normal(fcb1, 1024)
    # 使用relu函数来激活；
    fc1_R = tf.nn.relu(fcb1)
    # 进行dorpout,keepprob为出于活动状态的神经元百分比；
    # dropout1 = tf.nn.dropout(fc1, keepprob)
    # 进行第二次全连接层，神经元数量为4096
    weight2 = tf.Variable(tf.truncated_normal([1024, 1024], mean=0, stddev=0.01))
    # 定义全连接层的偏置值；
    biasesf2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1024]), trainable=True, name="biases")
    fc2 = tf.matmul(fc1_R, weight2) + biasesf2
    # nfc2 = batch_normal(fc2, 1024)
    fc2_R = tf.nn.relu(fc2)
    # 定义输出层，输出10个结果；
    weight3 = tf.Variable(tf.truncated_normal([1024, 10], mean=0, stddev=0.01))
    biasesf3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1024]), trainable=True, name="biases")
    # 同样使用sigmoid函数激活；
    fc3 = tf.nn.softmax(tf.matmul(fc2_R, weight3) + biasesf3)
    # 返回最终结果；
    return fc3


outputData = AlexNet(X_input)
# 损失函数；用交叉熵的方式定义；
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputData, labels=y_input))
# 定义指数下降学习率：
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, global_step, 20, 0.9, staircase=True)  # 生成学习率
# 进入优化器，用梯度下降的方式；
optimazer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
# 模型评估；
correct_pred = tf.equal(tf.argmax(outputData, 1), tf.argmax(y_input, 1))
# 求得一轮训练的平均得分；
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 初始化全部tf变量；
init = tf.global_variables_initializer()

# 创建一个会话；
with tf.Session() as sess:
    sess.run(init)
    c = []
    total_batch = int(X_train.shape[0]/batch_size)
    print(total_batch)
    for i in range(display_step):
        acc_add = 0
        for bantch in range(total_batch):
        # for bantch in range(1):
            start_time = time.time()
            batch_x = X_train[bantch*batch_size: (bantch + 1)*batch_size, :]
            batch_y = y_train[bantch * batch_size: (bantch + 1) * batch_size]
            a, b = sess.run([optimazer, loss], feed_dict={X_input: batch_x, y_input: batch_y})
            end_time = time.time()
            use_time = end_time - start_time
            acc = sess.run(accuracy, feed_dict={X_input: batch_x, y_input: batch_y})
            acc_add = acc_add + acc
            print("代价函数" + str(b) + "  训练批次：" + str(bantch) + "  训练时间：" + str(use_time) + "得分：" + str(acc))
        Main_Acc = acc_add/total_batch
        print("--------------------------------------------------------------------------第" + str(i) + "轮" + "本轮得分：" + str(Main_Acc))
    test = unpickle('cifar10-dataset/test_batch')
    X_test = test['data'][:5000, :]
    y_test = test['labels'][:5000]
    test_acc = sess.run(accuracy, feed_dict={X_input: X_test, y_input: y_test})
    print("测试集得分：" + str(test_acc))
