import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from CelebaLoader_64 import CelebaLoader
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ID = 'my_vae_celeba'
MODEL_PATH = './model/' + ID + '/'
OUT_PATH = './out/' + ID + '/'
LOG_PATH = './log/' + ID + '/'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

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


def weight_init(name, shape, num):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.5 / np.sqrt(num * 1.0)))


def D(x):
    with tf.variable_scope("D", reuse=tf.AUTO_REUSE):
        D_CONV1_W = weight_init("D_CONV1_W", [D_CONV1_SIZE, D_CONV1_SIZE, 3, D_CONV1_NUM], D_CONV1_SIZE * D_CONV1_SIZE)
        D_CONV1_B = tf.get_variable("D_CONV1_B", [D_CONV1_NUM], initializer=tf.zeros_initializer())
        D_CONV2_W = weight_init("D_CONV2_W", [D_CONV2_SIZE, D_CONV2_SIZE, D_CONV1_NUM, D_CONV2_NUM],
                                D_CONV2_SIZE * D_CONV2_SIZE)
        D_CONV2_B = tf.get_variable("D_CONV2_B", [D_CONV2_NUM], initializer=tf.zeros_initializer())
        D_CONV3_W = weight_init("D_CONV3_W", [D_CONV3_SIZE, D_CONV3_SIZE, D_CONV2_NUM, D_CONV3_NUM],
                                D_CONV3_SIZE * D_CONV3_SIZE)
        D_CONV3_B = tf.get_variable("D_CONV3_B", [D_CONV3_NUM], initializer=tf.zeros_initializer())
        D_CONV4_W = weight_init("D_CONV4_W", [D_CONV4_SIZE, D_CONV4_SIZE, D_CONV3_NUM, D_CONV4_NUM],
                                D_CONV4_SIZE * D_CONV4_SIZE)
        D_CONV4_B = tf.get_variable("D_CONV4_B", [D_CONV4_NUM], initializer=tf.zeros_initializer())
        D_W1 = weight_init("D_W1", [D_H1, D_H2], D_H1)
        D_B1 = tf.get_variable("D_B1", [D_H2], initializer=tf.zeros_initializer())
        D_MEAN_W1 = weight_init("D_MEAN_W1", [D_H2, D_H3], D_H2)
        D_MEAN_B1 = tf.get_variable("D_MEAN_B1", [D_H3], initializer=tf.zeros_initializer())
        D_STDDEV_W1 = weight_init("D_STDDEV_W1", [D_H2, D_H3], D_H2)
        D_STDDEV_B1 = tf.get_variable("D_STDDEV_B1", [D_H3], initializer=tf.zeros_initializer())

        tmp = tf.nn.leaky_relu(
            tf.nn.bias_add(tf.nn.conv2d(x, D_CONV1_W, [1, D_CONV1_STRIDE, D_CONV1_STRIDE, 1], "SAME"), D_CONV1_B))
        tmp = tf.nn.leaky_relu(
            tf.nn.bias_add(tf.nn.conv2d(tmp, D_CONV2_W, [1, D_CONV2_STRIDE, D_CONV2_STRIDE, 1], "SAME"), D_CONV2_B))
        tmp = tf.nn.leaky_relu(
            tf.nn.bias_add(tf.nn.conv2d(tmp, D_CONV3_W, [1, D_CONV3_STRIDE, D_CONV3_STRIDE, 1], "SAME"), D_CONV3_B))
        tmp = tf.nn.leaky_relu(
            tf.nn.bias_add(tf.nn.conv2d(tmp, D_CONV4_W, [1, D_CONV4_STRIDE, D_CONV4_STRIDE, 1], "SAME"), D_CONV4_B))
        tmp = tf.reshape(tmp, [-1, D_H1])
        tmp = tf.nn.leaky_relu(tf.matmul(tmp, D_W1) + D_B1)
        mean = tf.matmul(tmp, D_MEAN_W1) + D_MEAN_B1
        stddev = tf.matmul(tmp, D_STDDEV_W1) + D_STDDEV_B1
        return mean, stddev


def G(x, num):
    with tf.variable_scope("G", reuse=tf.AUTO_REUSE):
        G_H1_W = weight_init("G_H1_W", [G_IN, G_H1], G_IN)
        G_H1_B = tf.get_variable("G_H1_B", [G_H1], initializer=tf.zeros_initializer())
        G_H2_W = weight_init("G_H2_W", [G_H1, G_H2], G_H1)
        G_H2_B = tf.get_variable("G_H2_B", [G_H2], initializer=tf.zeros_initializer())
        G_CONV1_W = weight_init("G_CONV1_W", [G_CONV1_SIZE, G_CONV1_SIZE, G_CONV1_NUM, D_CONV4_NUM],
                                G_CONV1_SIZE * G_CONV1_SIZE)
        G_CONV1_B = tf.get_variable("G_CONV1_B", [G_CONV1_NUM], initializer=tf.zeros_initializer())
        G_CONV2_W = weight_init("G_CONV2_W", [G_CONV2_SIZE, G_CONV2_SIZE, G_CONV2_NUM, G_CONV1_NUM],
                                G_CONV2_SIZE * G_CONV2_SIZE)
        G_CONV2_B = tf.get_variable("G_CONV2_B", [G_CONV2_NUM], initializer=tf.zeros_initializer())
        G_CONV3_W = weight_init("G_CONV3_W", [G_CONV3_SIZE, G_CONV3_SIZE, G_CONV3_NUM, G_CONV2_NUM],
                                G_CONV3_SIZE * G_CONV3_SIZE)
        G_CONV3_B = tf.get_variable("G_CONV3_B", [G_CONV3_NUM], initializer=tf.zeros_initializer())
        G_CONV4_W = weight_init("G_CONV4_W", [G_CONV4_SIZE, G_CONV4_SIZE, G_CONV4_NUM, G_CONV3_NUM],
                                G_CONV4_SIZE * G_CONV4_SIZE)
        G_CONV4_B = tf.get_variable("G_CONV4_B", [G_CONV4_NUM], initializer=tf.zeros_initializer())
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


def M(z1, z2, num=BATCH_SIZE):
    with tf.variable_scope("M"):
        M_W = weight_init("M_W", [G_IN, G_IN], G_IN)
        M_B = tf.get_variable("M_B", [G_IN], initializer=tf.zeros_initializer())
        x1_dig,x1 = G(z1, num)
        z1_mean,z1_stddev = D(x1)
        x2_dig,x2 = G(z2, num)
        z2_mean,z2_stddev = D(x2)
        z_diff = tf.reduce_mean(tf.abs(z1_mean - z2_mean), 0)
        z_diff = tf.reshape(z_diff, [1, G_IN])
        tmp = tf.nn.bias_add(tf.matmul(z_diff, M_W), M_B)
    return tmp, tf.nn.softmax(tmp), [M_W, M_B]


global_step = tf.Variable(0, trainable=False)

X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 3])
MEAN, STDDEV = D(X)

Z = tf.truncated_normal([BATCH_SIZE, G_IN]) * STDDEV + MEAN
G_X_DIG, G_X = G(Z, BATCH_SIZE)
Z_TEST = tf.placeholder(tf.float32, [1, G_IN])
G_X_DIG1, G_X1 = G(Z_TEST, 1)

BUILD_LOSS = tf.reduce_mean(tf.square(X - G_X))
KL_LOSS = tf.reduce_mean(-1 - tf.log(tf.square(STDDEV) + 1e-12) + tf.square(STDDEV) + tf.square(MEAN))
LOSS = BUILD_LOSS * 100 + KL_LOSS * BETA
SOLVER = tf.train.AdamOptimizer(0.0001).minimize(LOSS, global_step=global_step)

M_Z1 = tf.placeholder(tf.float32, [None, G_IN])
M_Z2 = tf.placeholder(tf.float32, [None, G_IN])
M_Y = tf.placeholder(tf.float32, [1, G_IN])
M_RES_DIG, M_RES, M_VAR_LIST = M(M_Z1, M_Z2)
M_LOSS = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=M_Y, logits=M_RES_DIG))
M_SOLVER = tf.train.AdamOptimizer().minimize(M_LOSS, global_step, M_VAR_LIST)

LOADER = CelebaLoader("./Celeba/img_align_celeba/", "./Celeba/list_attr_celeba.txt")


def train_SOLVER(sess):
    x, _ = LOADER.read(BATCH_SIZE)
    _, loss, build_loss, kl_loss = sess.run([SOLVER, LOSS, BUILD_LOSS, KL_LOSS], feed_dict={X: x})
    return loss, build_loss, kl_loss, x


def train_M_SOLVER(sess):
    index = np.random.randint(0, 10)
    x, _ = LOADER.read(BATCH_SIZE)
    z1 = sess.run(Z, feed_dict={X: x})
    x, _ = LOADER.read(BATCH_SIZE)
    z2 = sess.run(Z, feed_dict={X: x})
    for i in range(len(z1)):
        z2[i][index] = z1[i][index]
    y = np.zeros([1, G_IN], dtype=np.float32)
    y[0][index] = 1.0
    _, m_loss = sess.run([M_SOLVER, M_LOSS], feed_dict={M_Z1: z1, M_Z2: z2, M_Y: y})
    return y, z1, z2, m_loss


tf.summary.scalar("build_loss", BUILD_LOSS)
tf.summary.scalar("m_loss", M_LOSS)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for it in range(100000):
        loss, build_loss, kl_loss, x = train_SOLVER(sess)
        y, z1, z2, m_loss = train_M_SOLVER(sess)
        step = sess.run(global_step)
        print('step {}:loss={:.4} buildloss={:.4} klloss={:.4} m_loss={:.4}'.format(step, loss, build_loss, kl_loss,
                                                                                    m_loss))
        if it % 100 == 0:
            summary = sess.run(summary_op, feed_dict={X: x, M_Y: y, M_Z1: z1, M_Z2: z2})
            writer.add_summary(summary, step)
            # figure = np.zeros((64 * 5, 64 * 10, 3))
            # z = np.reshape(sess.run(Z, feed_dict={X: x})[0], [1, G_IN])
            # for row in range(5):
            #     for col in range(10):
            #         z1 = np.array(z)
            #         z1[0][row] = 2.5 / 10.0 * col + 0.125
            #         x = sess.run(G_X1, feed_dict={Z_TEST: z1})
            #         figure[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64, :] = x
            #
            # fig = plt.figure(figsize=(20, 20))
            # plt.imshow(figure)
            # plt.savefig(OUT_PATH + '{}_y.png'.format(sess.run(global_step)), bbox_inches='tight')
            # plt.close(fig)
            saver.save(sess, MODEL_PATH, global_step=global_step)
    writer.close()

