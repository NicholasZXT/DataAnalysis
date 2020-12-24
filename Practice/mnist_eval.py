import tensorflow as tf
import time
import os
from tensorflow.examples.tutorials.mnist import input_data

from Practice import mnist_inference
from Practice import mnist_train

# 加载模型的间隔，5s一次
EVAL_INTERVAL_SECS = 5


def evaluate_dnn(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x_input')
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y_true')
        validate_feed = {x: mnist.validation.images, y_true: mnist.validation.labels}

        # 计算前向传播结果，这里不需要传入正则化项
        # 这个结果没有经过 softmax 层处理
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
                    print("经过 {} 轮训练后，模型在验证机上的精确度为 {}".format(global_step, accuracy_score))
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('./datasets/MNIST_data/', one_hot=True)
    evaluate_dnn(mnist)


if __name__ == '__main__':
    tf.app.run()
