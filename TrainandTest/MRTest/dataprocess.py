import copy
import numpy as np
import random
import mr
import utils
import os


# list_mr = ['Flipping', 'Transition', 'Scaling', 'Rotation', 'Noise', 'Color', 'Bright', 'RandomErasing' ]

def order_data_set(x_data, y_data):

    list_index = np.zeros((y_data.shape[1],), dtype=np.uint16)

    n_samples = x_data.shape[0]
    x_order = np.zeros((1, x_data.shape[1], x_data.shape[2], x_data.shape[3]), dtype=np.uint8)
    y_order = np.zeros((1, y_data.shape[1]), dtype=np.uint8)

    for i in range(n_samples):

        label = np.argmax(y_data[i], axis=0)

        x_order = np.insert(x_order, list_index[label], x_data[i], axis=0)
        y_order = np.insert(y_order, list_index[label], y_data[i], axis=0)

        if label < 9:
            list_index[label+1:] = list_index[label+1:]+1

        # print("i={}, label= {}".format(i, label))
        # print(list_index)
        # print(x_order.shape)

    x_order = np.delete(x_order, -1, axis=0)
    y_order = np.delete(y_order, -1, axis=0)

    return x_order, y_order, list_index


def sub_dataset(x_data_order, y_data_order, list_index, drop_labels):

    n_org_samples, n_org_labels = x_data_order.shape[0], y_data_order.shape[1]
    n_per_class = n_org_samples // n_org_labels
    print(n_org_samples, n_org_labels, n_per_class)

    for label in sorted(drop_labels, reverse=True):  # drop_label降序排列
        start = label*n_per_class
        if label == n_org_labels-1:
            end = n_org_samples
        else:
            end = start+n_per_class
        x_data_order = np.delete(x_data_order, slice(start, end), axis=0)
        y_data_order = np.delete(y_data_order, slice(start, end), axis=0)
        list_index = np.delete(list_index, -1)

    y_data_order = np.delete(y_data_order, drop_labels, axis=1)

    return x_data_order, y_data_order, list_index


def divide_train_test(x_order, y_order, list_index, n_per_tr, tr_over_te=5):
    n_per_te = n_per_tr//tr_over_te
    for start in list_index:
        if start == 0:
            x_train_order = copy.deepcopy(x_order[start: start+n_per_tr])
            y_train_order = copy.deepcopy(y_order[start: start + n_per_tr])
            x_test_order = copy.deepcopy(x_order[start+n_per_tr: start+n_per_tr+n_per_te])
            y_test_order = copy.deepcopy(y_order[start + n_per_tr: start + n_per_tr + n_per_te])
        else:
            x_train_order = np.concatenate((x_train_order, x_order[start: start+n_per_tr]), axis=0)
            y_train_order = np.concatenate((y_train_order, y_order[start: start+n_per_tr]), axis=0)
            x_test_order = np.concatenate((x_test_order, x_order[start+n_per_tr: start+n_per_tr+n_per_te]), axis=0)
            y_test_order = np.concatenate((y_test_order, y_order[start+n_per_tr: start+n_per_tr+n_per_te]), axis=0)

    return x_train_order, y_train_order, x_test_order, y_test_order


def type_a_dataset(x_data_order, y_data_order, list_index, n_per):
    n_sample = x_data_order.shape[0]
    n_labels = len(list_index)
    list_index_a = [0]*n_labels

    for i in range(n_labels):

        # 随机选取n_per个原样本
        if i == n_labels-1:
            start = random.randint(list_index[i], max(n_sample - n_per - 1, list_index[i]))
        else:
            # print(list_index[i], list_index[i + 1], n_per)
            start = random.randint(list_index[i], max(list_index[i + 1] - n_per - 1, list_index[i]))

        end = start + n_per
        # print('start', start, 'end', end)
        if i == 0:
            x_data = copy.deepcopy(x_data_order[start: end])
            y_data = copy.deepcopy(y_data_order[start: end])
        else:
            x_data = np.concatenate((x_data, x_data_order[start: end]), axis=0)
            y_data = np.concatenate((y_data, y_data_order[start: end]), axis=0)

            list_index_a[i] = list_index_a[i - 1] + n_per

    return x_data, y_data, list_index_a


def type_b_dataset(x_data_order, y_data_order, list_index, n_per, percent, alabel, mr_name, mr_par=None):
    """
    :param x_data_order: 按样本标签顺序排列的样本
    :param y_data_order: x_data_order中样本的标签
    :param list_index: x_data_order中每类样本的起始位置
    :param n_per: 每类样本的数量
    :param percent: 每类样本中原样本数量
    :param alabel: 含后续样本的样本标签
    :param mr_name: MR名字
    :return: 含后续样本的按标签顺序排列的样本及其标签
    """

    x_data_order, y_data_order, list_index = type_a_dataset(x_data_order, y_data_order, list_index, n_per)

    # 生成follow-up图片
    n_mr = n_per - int(n_per * percent)
    print('n_mr', n_mr)
    start = list_index[alabel]
    end = start + n_per
    x_data_al = copy.deepcopy(x_data_order[start:end])
    y_data_al = copy.deepcopy(y_data_order[start:end])
    x_data_mr, y_data_mr = mr.mr_output(copy.deepcopy(x_data_al[:n_mr]),
                                        copy.deepcopy(y_data_al[:n_mr]), mr_name, 1, mr_par)
    x_data_al[n_per - n_mr:] = x_data_mr
    y_data_al[n_per - n_mr:] = y_data_mr
    # 将x_data_al放置在x_data的最后
    x_data = np.delete(x_data_order, np.s_[start:start+n_per], axis=0)
    y_data = np.delete(y_data_order, np.s_[start:start+n_per], axis=0)
    x_data = np.concatenate((x_data, x_data_al), axis=0)
    y_data = np.concatenate((y_data, y_data_al), axis=0)

    return x_data, y_data


def type_c_dataset(x_data_order, y_data_order, list_index, n_per, percent, mr_name, mr_par=None):

    x_data_order, y_data_order, list_index = type_a_dataset(x_data_order, y_data_order, list_index, n_per)
    n_labels = len(list_index)
    n_mr = n_per - int(n_per * percent)

    for alabel in range(n_labels):

        # 生成follow-up图片
        start = list_index[alabel]
        end = start + n_mr
        x_data_al = copy.deepcopy(x_data_order[start:end])
        y_data_al = copy.deepcopy(y_data_order[start:end])
        x_data_mr, y_data_mr = mr.mr_output(x_data_al, y_data_al, mr_name, 1, mr_par)

        x_data_order[end:end+n_mr] = x_data_mr
        y_data_order[end:end+n_mr] = y_data_mr

    # x_data = (x_org_res) + x_org + x_mr

    return x_data_order, y_data_order


def type_d_dataset(x_data_order, y_data_order, list_index, n_per, percent, mr_name, mr_par=None):

    n_mr = int(n_per/percent*(1-percent))
    print('n_mr', n_mr)
    x_data_order, y_data_order, list_index = type_a_dataset(x_data_order, y_data_order, list_index, n_per)

    for i in range(len(list_index)):
        start = list_index[i]
        end = start + n_mr
        x_data_source = copy.deepcopy(x_data_order[start:end])
        y_data_source = copy.deepcopy(y_data_order[start:end])
        x_mr, y_mr = mr.mr_output(x_data_source, y_data_source, mr_name, 1, mr_par)

        x_data_order = np.concatenate((x_data_order, x_mr), axis=0)
        y_data_order = np.concatenate((y_data_order, y_mr), axis=0)

    return x_data_order, y_data_order


def type_x_dataset(type_x, x_data_order, y_data_order, list_index, input_par):

    if type_x == 'A':
        print("\n{} data set".format(type_x))
        return type_a_dataset(x_data_order, y_data_order, list_index, input_par.n_per_label)
    elif type_x == 'B':
        print("\n{} data set".format(type_x))
        return type_b_dataset(x_data_order, y_data_order, list_index, input_par.n_per_label,
                              input_par.percent, input_par.alabel, input_par.mr_name, input_par.mr_par)
    elif type_x == 'C':
        print("\n{} data set".format(type_x))
        return type_c_dataset(x_data_order, y_data_order, list_index, input_par.n_per_label,
                              input_par.percent, input_par.mr_name, input_par.mr_par)
    elif type_x == 'D':
        print("\n{} data set".format(type_x))
        return type_d_dataset(x_data_order, y_data_order, list_index, input_par.n_per_label,
                              input_par.percent, input_par.mr_name, input_par.mr_par)
    else:
        print("\n{} not Found.".format(type_x))
