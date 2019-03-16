import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 构建模型：
# input;
def get_inputs(real_size, noise_size):
    # 真实图像tensor与噪声图像张量
    real_img = tf.placeholder(tf.float32, [None, real_size], name='real_img')
    noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')
    return real_img, noise_img

# Generator:
def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    '''
    生成器；
    :param noise_img: 生成器输入；
    :param n_units: 隐藏层单元个数；
    :param out_dim: 输出张量的尺寸，MINST尺寸为：32*32=784
    :param reuse:
    :param alpha: leaky ReLU的系数
    :return:
    '''
    with tf.variable_scope('generator', reuse=reuse):
        # 隐藏层(使用一个全连接层；
        hidden1 = tf.layers.dense(noise_img, n_units)
        # leaky ReLU：
        hidden1 = tf.maximum(alpha*hidden1, hidden1)
        # dropout;
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)
        # logits & outputs
        logite = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logite)
        return logite, outputs

# Descriminator;
def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    '''
    辨别器
    :param img: 
    :param n_units: 隐藏层节点数量
    :param reuse: 
    :param alpha: LeakyReLU 系数；
    :return: 
    '''
    with tf.variable_scope('discriminitor', reuse=reuse):
        #  隐藏层；
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha*hidden1, hidden1)
        # logits & outputs
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)
        return  logits, outputs

# 定义参数；
# 真是图像的尺寸：
img_size = mnist.train.images[0].shape[0]
#  传入G 中的噪声size
noise_size = 100
# 生成器隐藏层参数
g_units = 128
# 辨别器隐藏层参数
d_units = 128
# leaky ReLU的参数；
alpha = 0.01
# 学习率
learning_rate = 0.01
# label smooothing
smooth = 0.1
# 构建网络；
tf.reset_default_graph()
# 输入；
real_img, noise_img =get_inputs(real_size=img_size, noise_size=noise_size)
# 生成器（G）
g_logits, g_outputs = get_generator(noise_img, g_units, img_size)
# 辨别器训练（D）
d_logits_real, d_outputs_real = get_discriminator(img=real_img, n_units=d_units)
# 辨别器进行辨别，输入生成器生成的图片；
d_logits_fake, d_outputs_fake = get_discriminator(img=g_outputs, n_units=d_units, reuse=True)
# loss:分别计算G,D的loss；
# discriminator:
# discriminator的目的在于对于给定的真图片，识别为真（1），对于generator生成的图片，识别为假（0），因此它的loss包含了真实图片的loss和生成器图片的loss两部分。
# 真实图片；
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real))*(1-smooth))
# 生成图片；
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
# d的总体loss:
d_loss = tf.add(d_loss_real, d_loss_fake)
# 生成器loss:
# generator generator的目的在于让discriminator识别不出它的图片是假的，如果用1代表真，0代表假，
# 那么generator生成的图片经过discriminator后要输出为1，因为generator想要骗过discriminator。
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_loss_fake, labels=tf.ones_like(d_loss_fake))*(1-smooth))
# 得到所有tf变量；
train_vars = tf.trainable_variables()
# generator:
g_vars = [var for var in train_vars if var.name.startswith('generator')]
# discriminator
d_vars = [var for var in train_vars if var.name.startswith('discriminitor')]
# Optimizer;
d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=g_loss, var_list=g_vars)

# 开始训练；
# batch_siza
batch_size = 64
# 迭代次数；
epochs = 300
# 抽取样本数量
n_sample = 25

# 存储样本的集合；
sample = []
# 存储损失函数的集合；
loss = []
# 保存模型对象；
saver = tf.train.Saver(var_list= g_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)

            batch_images = batch[0].reshape((batch_size, 784))
            # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
            batch_images = batch_images*2-1
            # generator 输入噪声
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
            # 运行Optimizers
            # 辨别器；
            _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img:batch_noise})
            # 生成器；
            _ = sess.run(g_train_opt,feed_dict={noise_img: batch_noise})
        # 每一轮结束后计算loss;
        train_loss_d = sess.run(d_loss, feed_dict={real_img:batch_images, noise_img:batch_noise})
