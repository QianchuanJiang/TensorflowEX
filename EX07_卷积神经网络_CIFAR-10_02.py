# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from CIFARreader import RotateImageFunc

# 读取并载入训练数据集的方法；
def unpickle(filename):
    with open(filename, 'rb') as fo:
        import pickle
        d = pickle.load(fo, encoding='latin1')
        return d

# 编写一个整数数字转换成OneHot的方法；
def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels
# print(onehot([0,5,8,6,8,10]))
# 开始导入训练集；
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
keepprob = 1.0

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

# 创建神经网络模型；构建模型参数；
training_iters = 200
batch_size = 1000
# 单个样本大小，图片为32x32像素，彩色3通道的图片，所以32x32x3=3072
n_features = 3072
# 标签种数；
n_classes = 10
# 第一全连接层数量；
n_fc1 = 384
# 第二全连接层数量；
n_fc2 = 192
# 构建训练特征和训练标签的占位符张量；
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])
# 定义权重张量，所有张量存储在一个字典中；结构是2层卷积层，2层全连接层，一个输出层；
W_conv = {
    'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.0001)),
    'conv2':  tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01)),
    'fc1': tf.Variable(tf.truncated_normal([8*8*64, n_fc1], stddev=0.1)),
    'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
    'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1)),
}
# 定义偏置值字典；结构同上；
b_conv = {
    'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32])),
    'conv2':  tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64])),
    'fc1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_fc1])),
    'fc2': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_fc2])),
    'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))
}
# 把特征值转换成图片格式的数组（4维张量）
X_image = tf.reshape(x, [-1, 32, 32, 3])
# 定义卷积层_1 步长为 1，卷积方式为SAME
conv1 = tf.nn.conv2d(X_image, W_conv['conv1'], strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
conv1 = tf.nn.relu(conv1)
# 定义池化层_1,池化单位大小为3x3，步长为2*2，池化方式为avg_pool,SAME方式；
pool1= tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# 定义lrn层
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# 定义第二层卷积层
conv2 = tf.nn.conv2d(norm1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
conv2 = tf.nn.relu(conv2)
# 定义lrn层
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
# 定义池化层_1,池化单位大小为3x3，步长为2*2，池化方式为avg_pool,SAME方式；
pool2 = tf.nn.avg_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# 把得到的数据进行扁平化处理后倒入到全连接层中；
reshape = tf.reshape(pool2, [-1, 8*8*64])
# 构建全连接层_1;
fc1 = tf.add(tf.matmul(reshape, W_conv['fc1']), b_conv['fc1'])
nfc1 = batch_normal(fc1, n_fc1)
fcr1 = tf.nn.relu(nfc1)
dropout1 = tf.nn.dropout(fc1, keepprob)
# 构建全连接层_2;
fc2 = tf.add(tf.matmul(dropout1, W_conv['fc2']), b_conv['fc2'])
nfc2 = batch_normal(fc2, n_fc2)
fcr2 = tf.nn.relu(nfc2)
dropout2 = tf.nn.dropout(fcr2, keepprob)
# 输出层（全连接_3）;
fc3 = tf.nn.softmax(tf.add(tf.matmul(dropout2, W_conv['fc3']), b_conv['fc3']))
# 损失函数；用交叉熵的方式定义；
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=y))
# 定义指数下降学习率：
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.03, global_step, 20, 0.9, staircase=True)  # 生成学习率
# 进入优化器，用梯度下降的方式；
optimazer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# 模型评估；
correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
# 求得一轮训练的平均得分；
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 初始化全部tf变量；
init = tf.global_variables_initializer()

# 创建一个会话；
with tf.Session() as sess:
    sess.run(init)
    AC = []
    COST = []
    total_batch = int(X_train.shape[0]/batch_size)
    print(total_batch)
    for i in range(training_iters):
        acc_add = 0
        for bantch in range(total_batch):
        # for bantch in range(1):
            start_time = time.time()
            batch_x = X_train[bantch*batch_size: (bantch + 1)*batch_size, :]
            batch_y = y_train[bantch*batch_size: (bantch + 1)*batch_size, :]
            batch_x_R = RotateImageFunc(batch_x)
            a, b = sess.run([optimazer, loss], feed_dict={x: batch_x_R, y: batch_y})
            end_time = time.time()
            use_time = end_time - start_time
            acc = sess.run(accuracy, feed_dict={x: batch_x_R, y: batch_y})
            acc_add = acc_add + acc
            AC.append(acc)
            COST.append(b)
            acc = acc*100
            print("损失函数：" + str(b) + "  训练批次：第" + str(i) + "循环，第： " + str(bantch) + "批   训练时间：" + str(use_time) + " 准确率：" + str(acc) + "%")
        Main_Acc = acc_add/total_batch
        Main_Acc = Main_Acc*100
        T = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("--------------------------------------------------------------------" + "时间：" + T + "第" + str(i) + "轮" + "本轮得分：" + str(Main_Acc) + "%")
    test_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
    test_acc = test_acc*100
    T = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("完成时间： " + T + "测试集得分：" + str(test_acc) + "%")

# 画图；
plt.plot(AC)
plt.plot(COST)
plt.xlabel('Iter')
plt.ylabel('Cost')
plt.title('Test_ACC: '+ str(test_acc) + '%')
plt.tight_layout()
plt.savefig('cnn-tf-cifar10-%s.png' % test_acc, dpi=1000)
