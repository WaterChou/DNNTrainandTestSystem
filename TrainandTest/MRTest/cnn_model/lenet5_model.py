import tensorflow as tf
import numpy as np
from collections import defaultdict


def LeNet5model(x, LayerOutput_flag=False, logits=False, training=False):

    layer_dict = defaultdict(float)    # dict，值类型为float

    # layer_dict['input'] = x

    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=6, kernel_size=[5, 5],
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv0_conv2d'] = z
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
        layer_dict['conv0_mp2d'] = z

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=16, kernel_size=[5, 5],
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv1_conv2d'] = z
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
        layer_dict['conv1_mp2d'] = z

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])
        # layer_dict['flatten'] = z

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=120, activation=tf.nn.relu)    # 全连接层
        z = tf.layers.dropout(z, rate=0.25, training=training)  # rate：每个元素被舍弃的概率
        layer_dict['mlp'] = z

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')
    layer_dict['logits'] = logits_
    # layer_dict['y'] = y

    if logits:
        if LayerOutput_flag:
            return y, logits_, layer_dict
        else:
            return y, logits_
    else:
        if LayerOutput_flag:
            return y, layer_dict
        else:
            return y
