import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # 只显示error
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import openpyxl as opxl

import models

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images.
"""


def unpickle(file_path):

    with open(file_path, 'rb') as fo:
        # data_dict = pickle.load(fo, encoding='bytes')  # official
        data_dict = pickle.load(fo, encoding='latin1')

    return data_dict


def shuffle_data(x_data, y_data):

    n_sample = x_data.shape[0]
    ind = np.arange(n_sample)
    np.random.shuffle(ind)
    x_data = x_data[ind]
    y_data = y_data[ind]

    return x_data, y_data


def read_mnist(readpath):
    """"
    读数据
    """
    print('\nLoading MNIST')
    img_size = 28
    img_chan = 1

    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    mnist = input_data.read_data_sets(readpath, one_hot=True)
    X_train = mnist.train.images
    X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
    y_train = mnist.train.labels

    X_valid = mnist.validation.images
    X_valid = np.reshape(X_valid, [-1, img_size, img_size, img_chan])
    y_valid = mnist.validation.labels

    X_test = mnist.test.images
    X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
    y_test = mnist.test.labels

    tf.logging.set_verbosity(old_v)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def read_cifar10_label_name(file_path):

    data_dict = unpickle(file_path)
    # dict_keys(['num_vis', 'label_names', 'num_cases_per_batch'])

    label_name = data_dict['label_names']  # len=10

    return label_name


def read_cifar10_train(train_path):

    for i in range(5):
        data_dict = unpickle(train_path[i])
        # dict_keys(['filenames', 'batch_label', 'data', 'labels'])
        x_data = data_dict['data']   # shape(10000, 3072)
        y_data = np.array(data_dict['labels'])  # shape(10000,)
        x_data = x_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # shape(10000, 32, 32, 3)

        if i == 0:
            x_train = x_data
            y_train = y_data
        else:
            x_train = np.concatenate((x_train, x_data), axis=0)
            y_train = np.concatenate((y_train, y_data), axis=0)

    return x_train, y_train


def read_cifar10_test(test_path):

    data_dict = unpickle(test_path)
    x_test = data_dict['data']  # shape(10000, 3072)
    y_test = np.array(data_dict['labels'])  # shape(10000,)
    x_test = x_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # shape(10000, 32, 32, 3)

    return x_test, y_test


def label_dense_to_one_hot(label_dense, class_num):

    n_sample = label_dense.shape[0]
    index_offset = np.arange(n_sample) * class_num  # 长度为n_sample的1D等差数组，起始值为0，步长为class_num
    label_one_hot = np.zeros(shape=(n_sample, class_num))
    label_one_hot.flat[index_offset + label_dense.ravel()] = 1  # flat:迭代器

    return label_one_hot


def calculate_graph(flag_mode_v, input_par):

    IMG_SIZE = input_par.img_size
    IMG_CHAN = input_par.img_chan
    CLASS_NUM = input_par.class_num

    class Dummy:  # 空类
        pass

    env = Dummy()  # 模型参数

    print('\nConstruction graph')

    with tf.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
        env.x = tf.placeholder(tf.float32, (None, IMG_SIZE, IMG_SIZE, IMG_CHAN), name='x')
        env.y = tf.placeholder(tf.float32, (None, CLASS_NUM), name='y')
        env.training = tf.placeholder(tf.bool, None, name='mode')  # 训练模式/非训练模式
        env.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        if flag_mode_v == 'lenet5':
            env.ybar, logits = models.model_v1(env.x, CLASS_NUM, logits=True,
                                               training=env.training)
        elif flag_mode_v == 'v2':
            env.ybar, logits = models.model_v2(env.x, CLASS_NUM, logits=True, training=env.training)
        elif flag_mode_v == 'vgg16':
            env.ybar, logits = models.model_vgg16(env.x, CLASS_NUM, logits=True, training=env.training)

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=env.y, logits=logits)
            env.loss = tf.reduce_mean(loss, name='loss')

        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(learning_rate=env.lr)
            env.train_op = optimizer.minimize(env.loss)

        env.saver = tf.train.Saver()  # 保存模型,默认保存全部变量
        # tf.get_default_graph().finalize()   # 只读图

    return env


def train(sess, env, x_data, y_data, x_valid=None, y_valid=None, epochs=1,
          lr_init=1e-4, load=False, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):   # 判断对象是否包含对应的属性
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        env.saver.restore(sess, 'model_save/{}'.format(name))

    print('\nTrain model')
    n_sample = x_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)

    for epoch in range(epochs):
        print('Epoch {0}/{1}'.format(epoch + 1, epochs))

        # shuffle
        x_data, y_data = shuffle_data(x_data, y_data)

        for batch in range(n_batch):
            # print('batch {0}/{1}'.format(batch + 1, n_batch), end='\r')

            start = int(batch * batch_size)
            end = min(n_sample, start + batch_size)

            lr = learning_rate_decay(lr_init, epoch)
            sess.run(env.train_op, feed_dict={env.x: x_data[start: end],
                                              env.y: y_data[start: end],
                                              env.lr: lr,
                                              env.training: True})

        if (x_valid is not None) and ((1+epoch) % 10 == 0):
            print('\nEvaluate on valid set')
            # print('lr', lr)
            loss, acc = evaluate(sess, env, x_valid, y_valid, batch_size=batch_size)
            print("loss: {0:.4f}, acc: {1:.4f}".format(loss, acc))
            if acc == 1/10:
                break

    #
    # if hasattr(env, 'saver'):
    #     print('\nSaving model')
    #     os.makedirs('model_save', exist_ok=True)
    #     env.saver.save(sess, 'model_save/{}'.format(name))


def evaluate(sess, env, x_data, y_data, batch_size=128, ):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    # print('\nEvaluating')

    n_sample = x_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc, batch_ybar = sess.run(
            [env.loss, env.acc, env.ybar],
            feed_dict={env.x: x_data[start:end],
                       env.y: y_data[start:end],
                       env.training: False})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample
    # print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))

    return loss, acc


def predict(sess, env, x_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = x_data.shape[0]  # shape[0]：第一维长度
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: x_data[start:end],
                                                env.training: False})
        yval[start:end] = y_batch

    print()
    return yval


def save_evaluate_res(sess, env, x_data, y_data, input_par_te, batch_size=128, flag_failed_label=False):

    print('\nSaving Evaluation res')

    n_sample = x_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc = 0, 0
    n_labels = input_par_te.class_num
    failed_label_n = np.zeros((n_labels,), dtype=np.uint)   # 被错分的各类标签样本数量
    failed_label = np.zeros(shape=(n_labels, n_labels), dtype=np.uint)  # 被错分的样本的输出

    for batch in range(n_batch):
        # print(' batch {0}/{1}'.format(batch + 1, n_batch))
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc, batch_ybar = sess.run(
            [env.loss, env.acc, env.ybar],
            feed_dict={env.x: x_data[start:end],
                       env.y: y_data[start:end],
                       env.training: False})
        loss += batch_loss * cnt
        acc += batch_acc * cnt

        for index in range(cnt):
            label = np.argmax(y_data[index + start], axis=0)
            prediction = np.argmax(batch_ybar[index], axis=0)

            if label != prediction:
                failed_label_n[label] += 1
                failed_label[label][prediction] += 1

    loss /= n_sample
    acc /= n_sample

    if flag_failed_label is False:

        return loss, acc, failed_label_n
    else:
        return loss, acc, failed_label_n, failed_label


def train_and_record(flag_mode_v, x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, epochs, lr_init,
                     res_path, input_par_tr, input_par_te, is_record=True):

    with tf.device('/gpu:2'):
        print('\nInitializing graph')
        env = calculate_graph(flag_mode_v, input_par_tr)

        sess = tf.InteractiveSession()

        sess.run(tf.global_variables_initializer())  # 执行了每个variable的initializer
        # sess.run(tf.local_variables_initializer())  # local_variables在图中并未被存储的变量

        tf.get_default_graph().finalize()  # 只读图，在训练过程中不增加新节点

        train(sess, env, x_train, y_train, x_valid, y_valid, load=False,
              lr_init=lr_init, epochs=epochs, batch_size=batch_size, name=flag_mode_v)

        if is_record is False:
            print('\nEvaluating on test set')
            loss, acc = evaluate(sess, env, x_test, y_test)
            print('\nloss={0:.4f},acc={1:.4f}'.format(loss, acc))

        else:
            print('\nSaving test res')
            print(res_path)

            loss, acc, failed_label_n = save_evaluate_res(sess, env,
                                                          x_test, y_test, input_par_te,
                                                          batch_size=batch_size)

            print('\nloss={0:.4f},acc={1:.4f}'.format(loss, acc))

            write_xlsx(res_path + 'loss_acc.xlsx', [loss, acc])
            # update_xlsx(res_path + 'n_failed_label.xlsx', list(failed_label_n))
            acc_labels = list(1 - failed_label_n / input_par_te.n_per_label)
            tmp = sorted(acc_labels)
            if tmp[0] != 0:
                write_xlsx(res_path + 'acc_label.xlsx', acc_labels)
            print(list(1 - failed_label_n / input_par_te.n_per_label))

    tf.reset_default_graph()
    sess.close()

    return acc


def learning_rate_decay(lr, epoch):
    # lr = 1e-4

    if epoch > 90:
        lr *= 1e-2
    elif epoch > 70:
        lr *= 1e-1
    if epoch > 30:
        lr *= 1e-1
    elif epoch > 40:
        lr *= 1e-1

    return lr


def new_xlsx(file_path):
    wb = opxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    wb.save(file_path)


def write_xlsx(file_path, list_input_data):

    if os.path.exists(file_path) is False:
        # print('New xlsx')
        new_xlsx(file_path)

    wb = opxl.load_workbook(filename=file_path)
    ws = wb['Sheet1']
    ws.append(list_input_data)
    wb.save(file_path)


def init_xlsx(res_path, input_par_tr, input_par_te, res_labels):
    if input_par_te.type == 'A':
        if os.path.exists(res_path) is False:
            os.makedirs(res_path, exist_ok=True)
            write_xlsx(res_path + 'loss_acc.xlsx', ['te percent', str(input_par_tr.percent)])
            write_xlsx(res_path + 'loss_acc.xlsx', ['loss', 'acc'])
        write_xlsx(res_path + 'acc_label.xlsx', ['te percent', str(input_par_tr.percent)])
        write_xlsx(res_path + 'acc_label.xlsx', res_labels)
    else:
        if os.path.exists(res_path) is False:
            os.makedirs(res_path, exist_ok=True)
            write_xlsx(res_path + 'loss_acc.xlsx', ['tr percent', str(input_par_tr.percent),
                                                    'te percent', str(input_par_te.percent)])
            write_xlsx(res_path + 'loss_acc.xlsx', ['loss', 'acc'])
            write_xlsx(res_path + 'acc_label.xlsx', ['tr percent', str(input_par_tr.percent),
                                                     'te percent', str(input_par_te.percent)])
            write_xlsx(res_path + 'acc_label.xlsx', res_labels)




