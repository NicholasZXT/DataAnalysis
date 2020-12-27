import tensorflow as tf
import time
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from Practice import mnist_inference
from Practice import mnist_train

# 加载模型的间隔，5s一次
EVAL_INTERVAL_SECS = 5


def evaluate(mnist, cnn=False):
    with tf.Graph().as_default() as g:
        valid_num = mnist.validation.images.shape[0]
        y_true = tf.placeholder(dtype=tf.float32, shape=[valid_num, mnist_inference.OUTPUT_NODE], name='y_true')
        if cnn:
            # CNN网络的输入
            x = tf.placeholder(dtype=tf.float32, shape=[valid_num, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                                                        mnist_inference.NUM_CHANNELS], name='x_input')

            validation_set_reshape = np.reshape(mnist.validation.images,(valid_num, mnist_inference.IMAGE_SIZE,
                                                        mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            validate_feed = {x: validation_set_reshape, y_true: mnist.validation.labels}
        else:
            # DNN网络的输入
            x = tf.placeholder(dtype=tf.float32, shape=[valid_num, mnist_inference.INPUT_NODE], name='x_input')
            validate_feed = {x: mnist.validation.images, y_true: mnist.validation.labels}

        # 计算前向传播结果，这里不需要传入正则化项
        # 这个结果没有经过 softmax 层处理
        if cnn:
            y_pred = mnist_inference.inference_lenet5(x, False, None)
        else:
            y_pred = mnist_inference.inference_dnn(x, None)

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(input=y_pred, axis=1), tf.argmax(input=y_true, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        # 下面这两句必须要加
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        while True:
            with tf.Session(config=config) as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名获取保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("经过 {} 轮训练后，模型在验证集上的精确度为 {:g}".format(global_step, accuracy_score))
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('./datasets/MNIST_data/', one_hot=True)
    # evaluate(mnist)
    evaluate(mnist, cnn=True)


if __name__ == '__main__':
    tf.app.run()
