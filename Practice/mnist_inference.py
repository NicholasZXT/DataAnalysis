import tensorflow as tf

# 神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_1_NODE = 500


def get_weights_variable(shape, regularizer):
    """
    获取权重，如果传入了正则化参数，则对权重进行正则化，并加入到 reg_loss 的集合里
    """
    weights = tf.get_variable(name="weigths", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection("reg_loss", regularizer(weights))
    return weights


def inference_dnn(input_tensor, regularizer):
    """
    神经网络的前向传播算法
    这里返回的 layer_2 没有进行 softmax 层的处理
    """
    with tf.variable_scope('layer1'):
        weights = get_weights_variable(shape=[INPUT_NODE, LAYER_1_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[LAYER_1_NODE], initializer=tf.constant_initializer(0.0))
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weights_variable(shape=[LAYER_1_NODE, OUTPUT_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights) + biases)
    return layer_2
