import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from CelebaLoader_64 import CelebaLoader
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ID = 'my_vae_celeba'
MODEL_PATH = './model/' + ID + '/'
OUT_PATH = './out/' + ID + '/'
LOSS_PATH = './loss/' + ID + '.txt'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists("./loss"):
    os.makedirs("./loss")

BETA = 1
BATCH_SIZE = 100
WIDTH = 64
HEIGHT = 64
D_IN = WIDTH * HEIGHT
D_CONV1_NUM = 32
D_CONV1_SIZE = 4
D_CONV1_STRIDE = 2
D_CONV2_NUM = 32
D_CONV2_SIZE = 4
D_CONV2_STRIDE = 2
D_CONV3_NUM = 64
D_CONV3_SIZE = 4
D_CONV3_STRIDE = 2
D_CONV4_NUM = 64
D_CONV4_SIZE = 4
D_CONV4_STRIDE = 2
D_H1 = 4 * 4 * D_CONV4_NUM
D_H2 = 256
D_H3 = 32

G_IN = 32
G_H1 = 256
G_H2 = 4 * 4 * D_CONV4_NUM
G_CONV1_NUM = D_CONV3_NUM
G_CONV1_SIZE = D_CONV4_SIZE
G_CONV1_STRIDE = D_CONV4_STRIDE
G_CONV2_NUM = D_CONV2_NUM
G_CONV2_SIZE = D_CONV3_SIZE
G_CONV2_STRIDE = D_CONV3_STRIDE
G_CONV3_NUM = D_CONV1_NUM
G_CONV3_SIZE = D_CONV2_SIZE
G_CONV3_STRIDE = D_CONV2_STRIDE
G_CONV4_NUM = 3
G_CONV4_SIZE = D_CONV1_SIZE
G_CONV4_STRIDE = D_CONV1_STRIDE


X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 3])
D_CONV1_W = tf.Variable(tf.truncated_normal([D_CONV1_SIZE, D_CONV1_SIZE, 3, D_CONV1_NUM], mean=0, stddev=0.25))
D_CONV1_B = tf.Variable(tf.zeros([D_CONV1_NUM]))
D_CONV2_W = tf.Variable(tf.truncated_normal([D_CONV2_SIZE, D_CONV2_SIZE, D_CONV1_NUM, D_CONV2_NUM], mean=0, stddev=0.25))
D_CONV2_B = tf.Variable(tf.zeros([D_CONV2_NUM]))
D_CONV3_W = tf.Variable(tf.truncated_normal([D_CONV3_SIZE, D_CONV3_SIZE, D_CONV2_NUM, D_CONV3_NUM], mean=0, stddev=0.25))
D_CONV3_B = tf.Variable(tf.zeros([D_CONV3_NUM]))
D_CONV4_W = tf.Variable(tf.truncated_normal([D_CONV4_SIZE, D_CONV4_SIZE, D_CONV3_NUM, D_CONV4_NUM], mean=0, stddev=0.25))
D_CONV4_B = tf.Variable(tf.zeros([D_CONV4_NUM]))
D_W1 = tf.Variable(tf.truncated_normal([D_H1, D_H2], mean=0, stddev=0.25))
D_B1 = tf.Variable(tf.zeros([D_H2]))
D_MEAN_W1 = tf.Variable(tf.truncated_normal([D_H2, D_H3], mean=0, stddev=0.25))
D_MEAN_B1 = tf.Variable(tf.zeros([D_H3]))
D_STDDEV_W1 = tf.Variable(tf.truncated_normal([D_H2, D_H3], mean=0, stddev=0.25))
D_STDDEV_B1 = tf.Variable(tf.zeros([D_H3]))

G_INPUT = tf.placeholder(tf.float32, [None, G_IN])
G_H1_W = tf.Variable(tf.truncated_normal([G_IN, G_H1], 0, 0.25))
G_H1_B = tf.Variable(tf.zeros([G_H1]))
G_H2_W = tf.Variable(tf.truncated_normal([G_H1, G_H2], 0, 0.25))
G_H2_B = tf.Variable(tf.zeros([G_H2]))
G_CONV1_W = tf.Variable(tf.truncated_normal([G_CONV1_SIZE, G_CONV1_SIZE, G_CONV1_NUM, D_CONV4_NUM], mean=0, stddev=0.25))
G_CONV1_B = tf.Variable(tf.zeros([G_CONV1_NUM]))
G_CONV2_W = tf.Variable(tf.truncated_normal([G_CONV2_SIZE, G_CONV2_SIZE, G_CONV2_NUM, G_CONV1_NUM], mean=0, stddev=0.25))
G_CONV2_B = tf.Variable(tf.zeros([G_CONV2_NUM]))
G_CONV3_W = tf.Variable(tf.truncated_normal([G_CONV3_SIZE, G_CONV3_SIZE, G_CONV3_NUM, G_CONV2_NUM], mean=0, stddev=0.25))
G_CONV3_B = tf.Variable(tf.zeros([G_CONV3_NUM]))
G_CONV4_W = tf.Variable(tf.truncated_normal([G_CONV4_SIZE, G_CONV4_SIZE, G_CONV4_NUM, G_CONV3_NUM], mean=0, stddev=0.25))
G_CONV4_B = tf.Variable(tf.zeros([G_CONV4_NUM]))


global_step = tf.Variable(0, trainable=False)


def D(x):
    tmp = tf.nn.leaky_relu(
        tf.nn.bias_add(tf.nn.conv2d(x, D_CONV1_W, [1, D_CONV1_STRIDE, D_CONV1_STRIDE, 1], "SAME"), D_CONV1_B))
    tmp = tf.nn.leaky_relu(
        tf.nn.bias_add(tf.nn.conv2d(tmp, D_CONV2_W, [1, D_CONV2_STRIDE, D_CONV2_STRIDE, 1], "SAME"), D_CONV2_B))
    tmp = tf.nn.leaky_relu(
        tf.nn.bias_add(tf.nn.conv2d(tmp, D_CONV3_W, [1, D_CONV3_STRIDE, D_CONV3_STRIDE, 1], "SAME"), D_CONV3_B))
    tmp = tf.nn.leaky_relu(
        tf.nn.bias_add(tf.nn.conv2d(tmp, D_CONV4_W, [1, D_CONV4_STRIDE, D_CONV4_STRIDE, 1], "SAME"), D_CONV4_B))
    tmp = tf.reshape(tmp, [-1, D_H1])
    tmp = tf.nn.leaky_relu(tf.matmul(tmp,D_W1)+D_B1)
    mean = tf.matmul(tmp, D_MEAN_W1) + D_MEAN_B1
    stddev = tf.matmul(tmp, D_STDDEV_W1) + D_STDDEV_B1
    return mean, stddev


def G(x, num):
    tmp = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(x, G_H1_W), G_H1_B))
    tmp = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(tmp, G_H2_W), G_H2_B))
    tmp = tf.reshape(tmp, [-1, 4, 4, D_CONV4_NUM])
    tmp = tf.nn.leaky_relu(tf.nn.bias_add(
        tf.nn.conv2d_transpose(tmp, G_CONV1_W, [num, 8, 8, G_CONV1_NUM], [1, G_CONV1_STRIDE, G_CONV1_STRIDE, 1]),
        G_CONV1_B))
    tmp = tf.nn.leaky_relu(tf.nn.bias_add(
        tf.nn.conv2d_transpose(tmp, G_CONV2_W, [num, 16, 16, G_CONV2_NUM], [1, G_CONV2_STRIDE, G_CONV2_STRIDE, 1]),
        G_CONV2_B))
    tmp = tf.nn.leaky_relu(tf.nn.bias_add(
        tf.nn.conv2d_transpose(tmp, G_CONV3_W, [num, 32, 32, G_CONV3_NUM], [1, G_CONV3_STRIDE, G_CONV3_STRIDE, 1]),
        G_CONV3_B))
    tmp = tf.nn.bias_add(
        tf.nn.conv2d_transpose(tmp, G_CONV4_W, [num, 64, 64, G_CONV4_NUM], [1, G_CONV4_STRIDE, G_CONV4_STRIDE, 1]),
        G_CONV4_B)
    return tmp, tf.nn.sigmoid(tmp)


MEAN, STDDEV = D(X)
Z = tf.truncated_normal([BATCH_SIZE, G_IN]) * STDDEV + MEAN
G_X_DIG, G_X = G(Z, BATCH_SIZE)

Z_TEST = tf.placeholder(tf.float32, [1, G_IN])
G_X_DIG1, G_X1 = G(Z_TEST, 1)


BUILD_LOSS = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=G_X_DIG))
KL_LOSS = tf.reduce_mean(-1 - tf.log(tf.square(STDDEV) + 1e-12) + tf.square(STDDEV) + tf.square(MEAN))
LOSS = BUILD_LOSS + KL_LOSS*BETA

SOLVER = tf.train.AdamOptimizer(0.0001).minimize(LOSS, global_step=global_step)
LOADER = CelebaLoader("./Celeba/img_align_celeba/", "./Celeba/list_attr_celeba.txt")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for it in range(1000000):
        x,_=LOADER.read(BATCH_SIZE)
        _, loss, build_loss, kl_loss = sess.run([SOLVER, LOSS, BUILD_LOSS, KL_LOSS],feed_dict={X: x})
        print('round {}:loss={:.4} buildloss={:.4} klloss={:.4}'.format(it, loss, build_loss, kl_loss))
        if it % 1000 == 0:
            # figure = np.zeros((64 * 5, 64 * 10, 3))
            # y=sess.run(G_X,feed_dict={X:x})
            # for row in range(5):
            #     for col in range(10):
            #         x=y[row*10+col]
            #         figure[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64, :] = x
            #
            # fig = plt.figure(figsize=(20, 20))
            # plt.imshow(figure)
            # plt.savefig(OUT_PATH + '{}_y.png'.format(sess.run(global_step)), bbox_inches='tight')
            # plt.close(fig)
            saver.save(sess, MODEL_PATH, global_step=global_step)
