import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from Practice import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZITATION_RATE = 0.0001
TRAINING_STEPS = 40000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./TensorFlow-Saved-Models/"
MODEL_NAME = "mnist_dnn_model.cpkt"


def train_dnn(mnist):
    x_batch = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x-batch')
    y_batch = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y-batch')

    regularizer = tf.nn.l2_loss
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARAZITATION_RATE)

    # 调用 mnist_inference 文件里的 inference 函数，获取 DNN 前向传播的结果——注意，这个结果没有经过 softmax 层处理
    y_pred = mnist_inference.inference_dnn(x_batch, regularizer)
    global_step = tf.Variable(0.0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                               decay_steps=mnist.train.num_examples / BATCH_SIZE,
                                               decay_rate=LEARNING_RATE_DECAY)

    # 计算预测向量和实际类别向量的交叉熵，在这之前，需要先对预测值进行一次 softmax 处理
    cross_entropy_vector = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_batch, logits=y_pred)
    # 计算交叉熵误差
    cross_entropy_loss = tf.reduce_mean(cross_entropy_vector)
    # 获取正则项
    reg_loss = tf.add_n(tf.get_collection("reg_loss"))
    # 总的误差项
    total_loss = cross_entropy_loss + REGULARAZITATION_RATE*reg_loss
    # 优化器和训练目标
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=total_loss,
                                                                                         global_step=global_step)
    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(input=y_pred, axis=1), tf.argmax(input=y_batch, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化操作
    init = tf.global_variables_initializer()
    # 模型持久化
    saver = tf.train.Saver()

    # 下面这两句必须要加
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        for step in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, train_loss, train_accuracy = sess.run([train_step, total_loss, accuracy], feed_dict={x_batch: xs, y_batch: ys})
            # 每 200 轮输出一次在验证集上的loss
            if (step + 1) % 200 == 0:
                print("经过 {} 轮训练后的在训练集上的交叉熵误差为：{:g}, 准确率为：{}".format(step + 1, train_loss, train_accuracy))
            # 每 5000 轮 保存一次模型参数
            if (step+1) % 5000 == 0:
                # 保存时，会在模型名称后面加上 global_step 的后缀
                saver.save(sess, MODEL_SAVE_PATH+MODEL_NAME, global_step=global_step)

    return ""


def main(argv=None):
    mnist = input_data.read_data_sets('./datasets/MNIST_data/', one_hot=True)
    train_dnn(mnist)


if __name__ == '__main__':
    tf.app.run()