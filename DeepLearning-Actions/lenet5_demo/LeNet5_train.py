import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

from . import LeNet5_inference

# 定义神经网络相关的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"


# 与第5章的区别在于，输入数据需要更改为四维
def train(mnist):
    # 1. 定义输入输出（参数在inference函数中）
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            LeNet5_inference.IMAGE_SIZE,
            LeNet5_inference.IMAGE_SIZE,
            LeNet5_inference.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_inference.OUTPUT_NODE], name='y-input')
    
    # 2. 定义前向传播、损失函数、反向传播
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_inference.inference(x,False,regularizer)
    
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(input=y_, axis=1), tf.argmax(input=y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    # 3. 建立会话，训练
    saver = tf.train.Saver()    # 初始化TensorFlow持久化类

    # 下面这两句必须要加
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                LeNet5_inference.IMAGE_SIZE,
                LeNet5_inference.IMAGE_SIZE,
                LeNet5_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            train_accuracy = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: ys})
            if i % 500 == 0:
                # print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                print("经过 {} 轮训练后的在训练集上的交叉熵误差为：{:g}, 准确率为：{:g}".format(step, loss_value, train_accuracy))
                # print("y_pred_max_index:\n",sess.run(tf.argmax(input=y, axis=1), feed_dict={x: reshaped_xs, y_: ys}))
                # print("y_pred_origin: \n", sess.run(y_pred, feed_dict={x_batch: xs, y_batch: ys}))
                # print("y_batch_max_index:\n",sess.run(tf.argmax(input=y_, axis=1), feed_dict={x: reshaped_xs, y_: ys}))
                # saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

                
                
def main(argv=None):
    mnist = input_data.read_data_sets('./datasets/MNIST_data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()