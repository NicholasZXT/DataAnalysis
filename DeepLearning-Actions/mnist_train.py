import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from Practice import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
# 这个参数的设置非常重要，太大的话，比如设置到0.8，就会导致效果非常差
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZITATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "TensorFlow-Saved-Models/"
TENSOR_BOARD_LOG_PATH = "TensorBoard-Logs"
MODEL_NAME = "MNIST_Model"


def train(mnist, cnn=False, save=False):
    """
    MNIST的神经网络训练函数，同时适用于DNN网络和CNN网络
    cnn=True表示使用CNN网络
    save表示是否保存模型
    """
    # 将输入的tensor放入 input-layer下，这是为了方便在Tensorboard中进行可视化
    with tf.name_scope("input-layer"):
        if cnn:
            x_batch = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                                                              mnist_inference.NUM_CHANNELS], name='x-batch')
        else:
            x_batch = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, mnist_inference.INPUT_NODE], name='x-batch')
        y_batch = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, mnist_inference.OUTPUT_NODE], name='y-batch')

    # L2 正则化函数
    regularizer = tf.nn.l2_loss
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARAZITATION_RATE)
    # 步长
    global_step = tf.Variable(0.0, trainable=False)

    # 调用 mnist_inference 文件里的 inference 函数，获取前向传播的结果——注意，这个结果没有经过 softmax 层处理
    if cnn:
        y_pred = mnist_inference.inference_lenet5(x_batch, train=True, regularizer=regularizer)
    else:
        y_pred = mnist_inference.inference_dnn(x_batch, regularizer)


    # 将损失函数等计算步骤放在loss_function命名空间下
    with tf.name_scope("loss_function"):
        # 计算预测向量和实际类别向量的交叉熵，在这之前，需要先对预测值进行一次 softmax 处理
        cross_entropy_vector = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_batch, logits=y_pred)
        # cross_entropy_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_batch, axis=1), logits=y_pred)
        # 计算交叉熵误差
        cross_entropy_loss = tf.reduce_mean(cross_entropy_vector)
        # 获取正则项形成的误差
        reg_loss = tf.add_n(tf.get_collection("reg_loss"))
        # 总的误差项
        total_loss = cross_entropy_loss + REGULARAZITATION_RATE*reg_loss
        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(input=y_pred, axis=1), tf.argmax(input=y_batch, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope("train_step"):
        # 指数衰减的学习率
        learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                                   decay_steps=mnist.train.num_examples / BATCH_SIZE,
                                                   decay_rate=LEARNING_RATE_DECAY)
        # 优化器和训练目标
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=total_loss,
                                                                                             global_step=global_step)

    # 初始化操作，这一步可以放到session里操作
    # init = tf.global_variables_initializer()
    # 模型持久化
    saver = tf.train.Saver()

    # 下面这两句必须要加
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # 初始化操作
        # sess.run(init)
        tf.global_variables_initializer().run()
        for step in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            if cnn:
                xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            _, train_loss, train_accuracy = sess.run([train_step, total_loss, accuracy], feed_dict={x_batch:xs, y_batch:ys})
            # 每 200 轮输出一次在验证集上的loss
            if step % 200 == 0:
                print("经过 {} 轮训练后的在训练集上的交叉熵误差为：{:g}, 准确率为：{:g}".format(step, train_loss, train_accuracy))
                # print("y_pred_max_index:\n", sess.run(tf.argmax(input=y_pred, axis=1), feed_dict={x_batch: xs, y_batch: ys}))
                # print("y_pred_origin: \n", sess.run(y_pred, feed_dict={x_batch: xs, y_batch: ys}))
                # print("y_batch_max_index:\n", sess.run(tf.argmax(input=y_batch, axis=1), feed_dict={x_batch: xs, y_batch: ys}))
                # print("y_batch_origin: \n", sess.run(y_batch, feed_dict={x_batch: xs, y_batch: ys}))
                # print("对比检查 :", sess.run(correct_prediction, feed_dict={x_batch: xs, y_batch: ys}))
            # 每 2000 轮 保存一次模型参数
            # if (step) % 2000 == 0 and save:
            if (step) % TRAINING_STEPS == 0 and save:
                # 保存时，会在模型名称后面加上 global_step 的后缀
                if cnn:
                    saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME + "-LeNet-5", global_step=global_step)
                else:
                    saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME + "-DNN", global_step=global_step)

    # 将当前的计算图输出到TensorBoard日志文件夹
    writer = tf.summary.FileWriter(TENSOR_BOARD_LOG_PATH, tf.get_default_graph())
    writer.close()

    return ""


def main(argv=None):
    mnist = input_data.read_data_sets('./datasets/MNIST_data/', one_hot=True)
    # train(mnist)
    # train(mnist, save=True)
    train(mnist, cnn=True)
    # train(mnist, cnn=True, save=True)


if __name__ == '__main__':
    tf.app.run()