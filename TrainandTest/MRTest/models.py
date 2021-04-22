import tensorflow as tf
import numpy as np
from collections import defaultdict


def model_v1(x, n_classes, logits=False, training=False):
    """
    LeNet5:CNN,2个卷积层，2个全连接层
    :param x: 输入样本
    :param logits: 是否输出logits
    :param training: 训练模式标识位
    :return: 预测值
    """
    layer_dict = defaultdict(float)

    with tf.compat.v1.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3], strides=(1, 1),
                                   padding='same', activation=tf.nn.relu, name='conv0_conv2d')
        layer_dict['conv0_conv2d'] = z
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same', name='conv0_maxpool')
        layer_dict['conv0_mp2d'] = z

    with tf.compat.v1.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3], strides=(1, 1),
                                   padding='same', activation=tf.nn.relu, name='conv1_conv2d')
        layer_dict['conv1_conv2d'] = z
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same', name='conv1_maxpool')
        layer_dict['conv1_mp2d'] = z

    with tf.compat.v1.variable_scope('conv2'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3], strides=(1, 1),
                                   padding='same', activation=tf.nn.relu, name='conv2_conv2d')
        layer_dict['conv2_conv2d'] = z
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same', name='conv2_maxpool')
        layer_dict['conv2_mp2d'] = z

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])  # np.prod() 连乘

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=1024, activation=tf.nn.relu)  # 全连接层
        z = tf.layers.dropout(z, rate=0.25, name='dense1', training=training)  # rate：每个元素被舍弃的概率
        layer_dict['mlp'] = z

    with tf.variable_scope('output'):
        logits_ = tf.layers.dense(z, units=n_classes, name='logits')
        y = tf.nn.softmax(logits_, name='ybar')
        layer_dict['mlp'] = z

    if logits:
        return y, logits_, layer_dict
    else:
        return y, layer_dict


def model_v2(x, n_classes, logits=False, training=False):

    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3], strides=(1, 1),
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same')

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3], strides=(1, 1),
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same')

    with tf.variable_scope('conv2'):
        z = tf.layers.conv2d(z, filters=256, kernel_size=[3, 3], strides=(1, 1),
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same')
        z = tf.layers.dropout(z, rate=0.25, training=training)  # rate：每个元素被舍弃的概率

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])  # np.prod() 连乘

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=1500, activation=tf.nn.relu)    # 全连接层 256*2*2
        z = tf.layers.dropout(z, rate=0.5, training=training)  # rate：每个元素被舍弃的概率

    with tf.variable_scope('output'):
        logits_ = tf.layers.dense(z, units=n_classes, name='logits')
        y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def model_vgg16(x, n_classes, logits=False, training=False):

    layer_dict = defaultdict(float)  # dict，值类型为float

    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv0_conv2d_1'] = z
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv0_conv2d_2'] = z
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same')
        layer_dict['conv0_mp2d'] = z

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.0),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv1_conv2d_1'] = z
        z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv1_conv2d_2'] = z
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same')
        layer_dict['conv1_mp2d'] = z

    with tf.variable_scope('conv2'):
        z = tf.layers.conv2d(z, filters=256, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.0),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv2_conv2d_1'] = z
        z = tf.layers.conv2d(z, filters=256, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.0),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv2_conv2d_2'] = z
        z = tf.layers.conv2d(z, filters=256, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same')
        layer_dict['conv2_mp2d'] = z

    with tf.variable_scope('conv3'):
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.0),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv3_conv2d_1'] = z
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.0),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv3_conv2d_2'] = z
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.0),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv3_conv2d_3'] = z
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same')
        layer_dict['conv3_mp2d'] = z

    with tf.variable_scope('conv4'):
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.0),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv4_conv2d_1'] = z
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.0),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv4_conv2d_2'] = z
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=(1, 1),
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.0),
                             padding='same', activation=tf.nn.relu)
        layer_dict['conv4_conv2d_3'] = z
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2, padding='same')
        layer_dict['conv4_mp2d'] = z

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])  # np.prod() 连乘

    with tf.variable_scope('fc6'):
        z = tf.layers.dense(z, units=4096, activation=tf.nn.relu)  # 全连接层
        z = tf.layers.dropout(z, rate=0.5, training=training)  # rate：每个元素被舍弃的概率
        layer_dict['fc6'] = z

    with tf.variable_scope('fc7'):
        z = tf.layers.dense(z, units=4096, activation=tf.nn.relu)  # 全连接层
        z = tf.layers.dropout(z, rate=0.5, training=training)  # rate：每个元素被舍弃的概率
        layer_dict['fc7'] = z

    with tf.variable_scope('output'):
        logits_ = tf.layers.dense(z, units=n_classes, name='logits')
        y = tf.nn.softmax(logits_, name='ybar')
        layer_dict['logits'] = logits_

    if logits:
        return y, logits_, layer_dict
    return y, layer_dict
