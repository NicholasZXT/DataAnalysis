import tensorflow as tf

# 神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
DNN_LAYER_1_NODE = 500
# 以下是CNN LeNet-5的参数
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
# 第一个卷积层的尺寸和深度
CONV_1_SZIE = 5
CONV_1_DEEP = 32
# 第二个卷积层的尺寸和深度
CONV_2_SZIE = 5
CONV_2_DEEP = 64
# 倒数第二个全连接层参数
FC_SIZE = 512


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
        weights = get_weights_variable(shape=[INPUT_NODE, DNN_LAYER_1_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[DNN_LAYER_1_NODE], initializer=tf.constant_initializer(0.0))
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('output_layer'):
        weights = get_weights_variable(shape=[DNN_LAYER_1_NODE, OUTPUT_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights) + biases)
    return layer_2


def inference_lenet5(input_tensor, train, regularizer):
    """
    使用LeNet-5经过稍微修改后的CNN网络
    同样，返回的结果也是没有经过softmax处理的
    使用了dropout和正则化
    train表示是否为训练过程，训练过程会使用dropout，预测过程则不需要
    1. 输入层：28x28x1
    2. 卷积层：32 个过滤器，步长为 1，**使用0填充**，卷积核为 5x5x1x32，输出尺寸为 28x28x32
    3. 池化层：窗口大小为 2x2，步长为 2，**使用0填充**，输出尺寸为 14x14x32
    4. 卷积层：64 个过滤器，步长为 1，不使用0填充，卷积核为 5x5x32x64，输出尺寸为 14-5+1=10，即 10x10x64
    5. 池化层：窗口大小为 2x2，步长为 2，输出尺寸为 5x5x64
    6. 全连接层：节点个数为 512
    7. 全连接层：节点个数为 10
    """
    with tf.variable_scope("layer1-conv-1"):
        # 输入为 28x28x1，卷积核为 5x5x1x32，输出尺寸为 28x28x32
        conv1_weights = tf.get_variable(name="weight", shape=[CONV_1_SZIE, CONV_1_SZIE, NUM_CHANNELS, CONV_1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 偏置个数=过滤器个数
        conv1_biases = tf.get_variable(name="bias", shape=[CONV_1_DEEP], initializer=tf.constant_initializer(0.0))
        # 卷积核步长=1，使用全0填充
        conv1 = tf.nn.conv2d(input=input_tensor, filter=conv1_weights, strides=1, padding="SAME")
        conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope("layer2-maxpool-1"):
        # 池化层窗口大小为 2x2，不使用全0填充
        pool_1 = tf.nn.max_pool2d(conv1_relu, ksize=2, strides=2, padding="VALID")
        # pool_1 = tf.nn.max_pool2d(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv-2"):
        # 输入为 14x14x32，卷积核为 5x5x32x64
        conv2_weights = tf.get_variable(name="weight", shape=[CONV_2_SZIE, CONV_2_SZIE, CONV_1_DEEP, CONV_2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(name="bias", shape=[CONV_2_DEEP], initializer=tf.constant_initializer(0.0))
        # 卷积核步长=1，不使用0填充
        conv2 = tf.nn.conv2d(input=pool_1, filter=conv2_weights, strides=1, padding="VALID")
        # conv2 = tf.nn.conv2d(input=pool_1, filter=conv2_weights, strides=1, padding="SAME")
        conv2_relu = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope("layer4-maxpool-2"):
        # 池化层窗口大小为 2x2,不使用全0填充
        pool_2 = tf.nn.max_pool2d(conv2_relu, ksize=2, strides=2, padding="VALID")
        # pool_2 = tf.nn.max_pool2d(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 上述得到的输出是 5x5x64，连接后续的全连接层之前，需要拉成一条向量
    pool_2_shape = pool_2.get_shape().as_list()
    # shape[0] 是batch_size
    vector_nodes = pool_2_shape[1] * pool_2_shape[2] * pool_2_shape[3]
    # 将第二个池化层的结果拉成一条向量
    pool_2_reshaped = tf.reshape(pool_2, shape=[pool_2_shape[0], vector_nodes])

    with tf.variable_scope("layer5-dense-1"):
        # 第一个全连接层，节点数为 512
        fc1_weights = tf.get_variable(name="weight", shape=[vector_nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只对全连接层的权重进行正则化
        if regularizer is not None:
            tf.add_to_collection("reg_loss", regularizer(fc1_weights))
        fc1_bias = tf.get_variable(name='bias', shape=[FC_SIZE], initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(pool_2_reshaped, fc1_weights) + fc1_bias)
        # train表示是否加入dropout，一般只在全连接层加入，并且只在训练时加入
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("layer6-dense-2"):
        # 第二个，也是最后一个全连接层，节点数 = 10，也就是分类数
        fc2_weights = tf.get_variable(name="weight", shape=[FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 同样，只对全连接层权重进行正则化
        if regularizer is not None:
            tf.add_to_collection("reg_loss", regularizer(fc2_weights))
        fc2_bias = tf.get_variable(name='bias', shape=[NUM_LABELS], initializer=tf.constant_initializer(0.0))
        y_pred = tf.matmul(fc1, fc2_weights) + fc2_bias
    # 最终输出的y_pred是 bacth_size x 10
    return y_pred


def inference_inception(mnist):
    pass
