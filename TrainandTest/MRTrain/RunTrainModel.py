import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy

import utils
import dataprocess as dp
from my_class import InputPar


class RunDataset:

    def __init__(self, config_dict):

        self.dataset = config_dict['dataset']
        self.flag_train_mr = config_dict['type_name']  # 'A', 'B', 'C'

        self.BATCH_SIZE = 100
        self.EPOCHS = config_dict['epochs']  # 50-k=2500, 150-k=5000
        self.LR_init = config_dict['lr']
        self.RUN_TIMES = config_dict['runtime']
        self.model_name = config_dict['model']
        self.k = config_dict['k']
        self.mr_per = config_dict['percent']
        self.mr_name = config_dict['mr_name']
        if config_dict['type_name'] == 'B':
            self.alabel = config_dict['alabel']
        self.flag_save_data = config_dict['save']
        self.drop_label = config_dict['drop_label']
        self.res_label = list()

        self.file_path = config_dict['file_name']

        self.acc = 0

    def get_cifar10_data(self):
        cifar10_label_name_path = '{}CIFAR-10/batches.meta'.format(self.file_path)
        cifar10_test_data_path = '{}CIFAR-10/test_batch'.format(self.file_path)

        cifar10_train_data_path = ['{}CIFAR-10/data_batch_1'.format(self.file_path),
                                   '{}CIFAR-10/data_batch_2'.format(self.file_path),
                                   '{}CIFAR-10/data_batch_3'.format(self.file_path),
                                   '{}CIFAR-10/data_batch_4'.format(self.file_path),
                                   '{}CIFAR-10/data_batch_5'.format(self.file_path)]

        print('\nLoading CIFAR-10')

        x_train, y_train = utils.read_cifar10_train(cifar10_train_data_path)
        x_test, y_test = utils.read_cifar10_test(cifar10_test_data_path)
        label_name_list = utils.read_cifar10_label_name(cifar10_label_name_path)
        # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        print(label_name_list)
        self.IMG_SIZE = x_train.shape[1]
        self.IMG_CHAN = x_train.shape[-1]
        self.CLASS_NUM = len(label_name_list)

        y_train = utils.label_dense_to_one_hot(y_train, self.CLASS_NUM)
        y_test = utils.label_dense_to_one_hot(y_test, self.CLASS_NUM)

        flag_order_train_load = True
        flag_order_test_load = True

        # train_order
        order_data_path = "{}order_data/CIFAR10/".format(self.file_path)
        if os.path.exists('{}x_train_order.npy'.format(order_data_path)) and flag_order_train_load:
            print('\nLoad Train order data')
            x_train_order = np.load('{}x_train_order.npy'.format(order_data_path))
            y_train_order = np.load('{}y_train_order.npy'.format(order_data_path))
            list_index_train = np.load('{}list_index_train.npy'.format(order_data_path))
        else:
            print('\nOrdering...')
            x_train_order, y_train_order, list_index_train = dp.order_data_set(x_train, y_train)
            os.makedirs(order_data_path, exist_ok=True)
            np.save('{}x_train_order.npy'.format(order_data_path), x_train_order)
            np.save('{}y_train_order.npy'.format(order_data_path), y_train_order)
            np.save('{}list_index_train.npy'.format(order_data_path), list_index_train)

        # print('x_tr_order shape = {0}. y_tr_order shape = {1}'.format(x_train_order.shape, y_train_order.shape))
        # print(list_index_train)

        for i in range(self.CLASS_NUM):
            if i in self.drop_label:
                self.res_label.append(i)
        self.res_label.sort()

        if len(self.drop_label) > 0:
            self.x_train_order, self.y_train_order, self.list_index_train = dp.sub_dataset(x_train_order, y_train_order,
                                                                                            list_index_train,
                                                                                            self.drop_label)
            # print('x_tr_order_sub shape = {0}. y_tr_order_sub shape = {1}'.format(self.x_train_order.shape,
            #                                                                       self.y_train_order.shape))
            # print(self.list_index_train)
        else:
            self.x_train_order = x_train_order
            self.y_train_order = y_train_order
            self.list_index_train = list_index_train

        # test_order
        if os.path.exists('{}x_test_order.npy'.format(order_data_path)) and flag_order_test_load:
            print('\nLoad Test order data')
            x_test_order = np.load('{}x_test_order.npy'.format(order_data_path))
            y_test_order = np.load('{}y_test_order.npy'.format(order_data_path))
            list_index_test = np.load('{}list_index_test.npy'.format(order_data_path))
        else:
            print('\nOrdering...')
            x_test_order, y_test_order, list_index_test = dp.order_data_set(x_test, y_test)
            os.makedirs(order_data_path, exist_ok=True)
            np.save('{}x_test_order.npy'.format(order_data_path), x_test_order)
            np.save('{}y_test_order.npy'.format(order_data_path), y_test_order)
            np.save('{}list_index_test.npy'.format(order_data_path), list_index_test)

        for i in range(self.CLASS_NUM):
            if i not in self.drop_label:
                self.res_label.append(i)
        self.res_label.sort()

        if len(self.drop_label) > 0:
            self.x_test_order, self.y_test_order, self.list_index_test = dp.sub_dataset(x_test_order, y_test_order,
                                                                                        list_index_test,
                                                                                        self.drop_label)
        else:
            self.x_test_order = x_test_order
            self.y_test_order = y_test_order
            self.list_index_test = list_index_test

        # print('x_te_order shape = {0}. y_te_order shape = {1}'.format(self.x_test_order.shape, self.y_test_order.shape))
        # print(self.list_index_test)

    def get_mnist_data(self):

        order_data_path = "{}order_data/MNIST/".format(self.file_path)
        x_train, y_train, x_valid, y_valid, x_test, y_test = utils.read_mnist("{}MNIST_data/".format(order_data_path))  # 读数据

        self.IMG_SIZE = x_train.shape[1]
        self.IMG_CHAN = x_train.shape[-1]
        self.CLASS_NUM = y_train.shape[-1]
        # print('img_size', self.IMG_SIZE)

        x_data = np.concatenate((x_train, x_test, x_valid), axis=0)
        y_data = np.concatenate((y_train, y_test, y_valid), axis=0)

        flag_order_load = True
        # data_order
        print('\n{}x_order.npy'.format(order_data_path))
        print(os.path.isfile('{}x_order.npy'.format(order_data_path)))
        if os.path.isfile('{}x_order.npy'.format(order_data_path)) and flag_order_load:
            print('\nLoad Train order data')
            x_order_org = np.load('{}x_order.npy'.format(order_data_path))
            y_order_org = np.load('{}y_order.npy'.format(order_data_path))
            list_index_org = np.load('{}list_index.npy'.format(order_data_path))
        else:
            print('\nOrdering...')
            x_order_org, y_order_org, list_index_org = dp.order_data_set(x_data, y_data)
            os.makedirs(order_data_path, exist_ok=True)
            np.save('{}x_order.npy'.format(order_data_path), x_order_org)
            np.save('{}y_order.npy'.format(order_data_path), y_order_org)
            np.save('{}list_index.npy'.format(order_data_path), list_index_org)

        # print(list_index_org)

        x_train_order_org, y_train_order_org, x_test_order_org, y_test_order_org \
            = dp.divide_train_test(x_order_org, y_order_org, list_index_org, n_per_tr=5000, tr_over_te=5)

        list_index_train_org = list(range(0, 5000 * self.CLASS_NUM, 5000))
        list_index_test_org = list(range(0, 1000 * self.CLASS_NUM, 1000))
        # print(x_train_order_org.shape)
        # print(list_index_train_org)

        # print("drop label", self.drop_label)
        for i in range(self.CLASS_NUM):
            if i not in self.drop_label:
                self.res_label.append(i)
        self.res_label.sort()
        # print(print("res label", self.res_label))

        if len(self.drop_label) > 0:
            self.x_train_order, self.y_train_order, self.list_index_train = dp.sub_dataset(x_train_order_org,
                                                                                           y_train_order_org,
                                                                                           list_index_train_org,
                                                                                           self.drop_label)
        else:
            self.x_train_order = x_train_order_org
            self.y_train_order = y_train_order_org
            self.list_index_train = list_index_train_org

        # print('x_tr_order_sub shape = {0}. y_tr_order_sub shape = {1}'.format(self.x_train_order.shape,
        #                                                                       self.y_train_order.shape))
        # print(self.list_index_train)

        if len(self.drop_label) > 0:

            self.x_test_order, self.y_test_order, self.list_index_test = dp.sub_dataset(x_test_order_org,
                                                                                        y_test_order_org,
                                                                                        list_index_test_org,
                                                                                        self.drop_label)
        else:
            self.x_test_order = x_test_order_org
            self.y_test_order = y_test_order_org
            self.list_index_test = list_index_test_org

        # print('x_te_order_sub shape = {0}. y_te_order_sub shape = {1}'.format(self.x_test_order.shape,
        #                                                                       self.y_test_order.shape))
        # print(self.list_index_test)

    def run(self):

        if self.dataset == 'CIFAR10':
            self.get_cifar10_data()
        else:
            self.get_mnist_data()

        input_par_te = InputPar.InputPar('A', self.IMG_SIZE, self.IMG_CHAN, len(self.res_label),
                                         n_per_label=self.k // 5)
        x_test, y_test, _ = dp.type_x_dataset('A', self.x_test_order, self.y_test_order,
                                              self.list_index_test, input_par_te)

        if self.flag_train_mr == 'A':
            print('\nType A Training set')
            input_par_tr = InputPar.InputPar(self.flag_train_mr, self.IMG_SIZE, self.IMG_CHAN, len(self.res_label),
                                             n_per_label=self.k)  # 1-0.25, 1-0.12, 1-0.06
            print(len(self.res_label))

            res_path = self.file_path + 'test_res/' + self.dataset + '/A_k' + str(input_par_tr.n_per_label) + '/'
            utils.init_xlsx(res_path, input_par_tr, input_par_te, self.res_label)

            for n in range(self.RUN_TIMES):
                print('\nRun times = {}/{}'.format(n + 1, self.RUN_TIMES))
                time_start = time.time()

                x_train, y_train, _ = dp.type_x_dataset('A', self.x_train_order, self.y_train_order,
                                                        self.list_index_train,
                                                        input_par_tr)
                print('\nx_train shape = {0}. y_train shape = {1}'.format(x_train.shape, y_train.shape))

                print('EPOCHS', self.EPOCHS)
                print(self.model_name)
                self.acc = utils.train_and_record(self.model_name, x_train, y_train,
                                                  x_train, y_train, x_test, y_test,
                                                  lr_init=self.LR_init, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS,
                                                  res_path=res_path,
                                                  input_par_tr=input_par_tr, input_par_te=input_par_te,
                                                  is_record=self.flag_save_data)

                time_end = time.time()

                print('\nTime_len={}'.format(time_end - time_start))

        else:
            input_par_tr = InputPar.InputPar(self.flag_train_mr, self.IMG_SIZE, self.IMG_CHAN, len(self.res_label),
                                             n_per_label=self.k,
                                             percent=1 - self.mr_per)  # 1-0.25, 1-0.12, 1-0.06
            input_par_tr.mr_name = self.mr_name
            input_par_tr.mr_par = None
            mr_info = input_par_tr.mr_name + str(input_par_tr.mr_par)

            if self.flag_train_mr == 'B':
                print('Type B Training set')
                input_par_tr.n_follow_up_samples = int((1 - input_par_tr.percent) * input_par_tr.n_per_label)

                input_par_tr.alabel = self.alabel

                res_path = self.file_path + '/test_res/' +self.dataset + '/percent' + str(self.mr_per) + \
                           '/B_' + mr_info + '_label' + str(self.res_label[self.alabel]) \
                           + '_k' + str(input_par_tr.n_per_label) + '/'
                utils.init_xlsx(res_path, input_par_tr, input_par_te, self.res_label)

                for n in range(self.RUN_TIMES):
                    print('\nRun times = {}/{}. MR={}. \nAL={}. k={}'.format(n + 1, self.RUN_TIMES, mr_info,
                                                                             self.res_label[self.alabel],
                                                                             input_par_tr.n_per_label))

                    x_train, y_train = dp.type_x_dataset('B', self.x_train_order, self.y_train_order,
                                                         self.list_index_train,
                                                         input_par_tr)

                    time_start = time.time()

                    self.acc = utils.train_and_record(self.model_name, x_train, y_train, x_train, y_train,
                                                      x_test, y_test,
                                                      self.BATCH_SIZE, self.EPOCHS, self.LR_init,
                                                      res_path, input_par_tr, input_par_te,
                                                      is_record=self.flag_save_data)

                    time_end = time.time()
                    print('\nTime_len={}'.format(time_end - time_start))

            elif self.flag_train_mr == 'C':
                print('Type C Training set')

                input_par_tr.n_follow_up_samples = \
                    int(input_par_tr.percent * input_par_tr.n_per_label) * len(self.res_label)

                res_path = self.file_path + '/test_res/' + self.dataset + '/percent' + str(self.mr_per) + \
                           '/C_' + mr_info + '_k' + str(input_par_tr.n_per_label) + '/'
                utils.init_xlsx(res_path, input_par_tr, input_par_te, self.res_label)

                for n in range(self.RUN_TIMES):
                    print('\nRun times = {}/{}. MR={}. \nk={}'.format(n + 1, self.RUN_TIMES,
                                                                      mr_info, input_par_tr.n_per_label))

                    x_train, y_train = dp.type_x_dataset('C', self.x_train_order, self.y_train_order,
                                                         self.list_index_train,
                                                         input_par_tr)

                    time_start = time.time()

                    self.acc = utils.train_and_record(self.model_name, x_train, y_train, None, None, x_test, y_test,
                                                      self.BATCH_SIZE, self.EPOCHS, self.LR_init,
                                                      res_path, input_par_tr, input_par_te,
                                                      is_record=self.flag_save_data)

                    time_end = time.time()
                    print('\nTime_len={}'.format(time_end - time_start))

            else:
                print('Not found Type {} Training set'.format(self.flag_train_mr))


def cifar10(config_dict):

    cifar10_label_name_path = './CIFAR-10/batches.meta'
    cifar10_test_data_path = './CIFAR-10/test_batch'

    cifar10_train_data_path = ['./CIFAR-10/data_batch_1',
                               './CIFAR-10/data_batch_2',
                               './CIFAR-10/data_batch_3',
                               './CIFAR-10/data_batch_4',
                               './CIFAR-10/data_batch_5']

    print('\nLoading CIFAR-10')

    x_train, y_train = utils.read_cifar10_train(cifar10_train_data_path)
    x_test, y_test = utils.read_cifar10_test(cifar10_test_data_path)
    label_name_list = utils.read_cifar10_label_name(cifar10_label_name_path)
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(label_name_list)
    IMG_SIZE = x_train.shape[1]
    IMG_CHAN = x_train.shape[-1]
    CLASS_NUM = len(label_name_list)

    y_train = utils.label_dense_to_one_hot(y_train, CLASS_NUM)
    y_test = utils.label_dense_to_one_hot(y_test, CLASS_NUM)

    flag_order_train_load = True
    flag_order_test_load = True

    # train_order
    if os.path.exists('./order_data/x_train_order.npy') and flag_order_train_load:
        print('\nLoad Train order data')
        x_train_order = np.load('./order_data/x_train_order.npy')
        y_train_order = np.load('./order_data/y_train_order.npy')
        list_index_train = np.load('./order_data/list_index_train.npy')
    else:
        print('\nOrdering...')
        x_train_order, y_train_order, list_index_train = dp.order_data_set(x_train, y_train)
        os.makedirs('./order_data/', exist_ok=True)
        np.save('./order_data/x_train_order.npy', x_train_order)
        np.save('./order_data/y_train_order.npy', y_train_order)
        np.save('./order_data/list_index_train.npy', list_index_train)

    print('x_tr_order shape = {0}. y_tr_order shape = {1}'.format(x_train_order.shape, y_train_order.shape))
    print(list_index_train)

    drop_labels = config_dict['drop_labels']
    res_labels = list(set(np.arange(0, CLASS_NUM, 1))-set(drop_labels))
    res_labels.sort()
    if len(drop_labels) > 0:
        x_train_order, y_train_order, list_index_train = dp.sub_dataset(x_train_order, y_train_order, list_index_train,
                                                                        drop_labels)
        print('x_tr_order_sub shape = {0}. y_tr_order_sub shape = {1}'.format(x_train_order.shape, y_train_order.shape))
        print(list_index_train)

    # test_order
    if os.path.exists('./order_data/x_test_order.npy') and flag_order_test_load:
        print('\nLoad Test order data')
        x_test_order = np.load('./order_data/x_test_order.npy')
        y_test_order = np.load('./order_data/y_test_order.npy')
        list_index_test = np.load('./order_data/list_index_test.npy')
    else:
        print('\nOrdering...')
        x_test_order, y_test_order, list_index_test = dp.order_data_set(x_test, y_test)
        os.makedirs('./order_data/', exist_ok=True)
        np.save('./order_data/x_test_order.npy', x_test_order)
        np.save('./order_data/y_test_order.npy', y_test_order)
        np.save('./order_data/list_index_test.npy', list_index_test)

    if len(drop_labels) > 0:
        x_test_order, y_test_order, list_index_test = dp.sub_dataset(x_test_order, y_test_order, list_index_test,
                                                                     drop_labels)

    print('x_te_order shape = {0}. y_te_order shape = {1}'.format(x_test_order.shape, y_test_order.shape))
    print(list_index_test)

    flag_train_mr = config_dict['type_name']   # 'A', 'B', 'C'

    BATCH_SIZE = 100
    EPOCHS = config_dict['epochs']  # 50-k=2500, 150-k=5000
    LR_init = config_dict['lr']
    RUN_TIMES = config_dict['runtime']
    model_name = config_dict['model']
    k = config_dict['k']
    mr_per = config_dict['percent']
    mr_name = config_dict['mr_name']
    alabel = config_dict['alabel']
    flag_save_data = config_dict['save']

    input_par_te = InputPar.InputPar('A', IMG_SIZE, IMG_CHAN, len(res_labels),
                                     n_per_label=k // 5)
    x_test, y_test, _ = dp.type_x_dataset('A', x_test_order, y_test_order, list_index_test, input_par_te)

    if flag_train_mr == 'A':
        print('\nType A Training set')
        input_par_tr = InputPar.InputPar(flag_train_mr, IMG_SIZE, IMG_CHAN, len(res_labels),
                                         n_per_label=k)  # 1-0.25, 1-0.12, 1-0.06

        res_path = './test_res/A_k' + str(input_par_tr.n_per_label) + '/'
        utils.init_xlsx(res_path, input_par_tr, input_par_te, res_labels)

        for n in range(RUN_TIMES):

            print('\nRun times = {}/{}'.format(n + 1, RUN_TIMES))
            time_start = time.time()

            x_train, y_train, _ = dp.type_x_dataset('A', x_train_order, y_train_order, list_index_train, input_par_tr)
            print('\nx_train shape = {0}. y_train shape = {1}'.format(x_train.shape, y_train.shape))

            print('EPOCHS', EPOCHS)
            utils.train_and_record(model_name, x_train, y_train, x_train, y_train, x_test, y_test,
                                   lr_init=LR_init, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                   res_path=res_path,
                                   input_par_tr=input_par_tr, input_par_te=input_par_te, is_record=flag_save_data)

            time_end = time.time()

            print('\nTime_len={}'.format(time_end - time_start))

    else:

        input_par_tr = InputPar.InputPar(flag_train_mr, IMG_SIZE, IMG_CHAN, len(res_labels),
                                         n_per_label=k,
                                         percent=1-mr_per)   # 1-0.25, 1-0.12, 1-0.06
        input_par_tr.mr_name = mr_name
        input_par_tr.mr_par = None
        mr_info = input_par_tr.mr_name + str(input_par_tr.mr_par)

        if flag_train_mr == 'B':
            print('Type B Training set')
            input_par_tr.n_follow_up_samples = int((1-input_par_tr.percent)*input_par_tr.n_per_label)

            input_par_tr.alabel = alabel

            res_path = './test_res/percent'+str(mr_per) + \
                       '/B_' + mr_info + '_label' + str(res_labels[alabel]) \
                       + '_k' + str(input_par_tr.n_per_label) + '/'
            utils.init_xlsx(res_path, input_par_tr, input_par_te, res_labels)

            for n in range(RUN_TIMES):
                print('\nRun times = {}/{}. MR={}. \nAL={}. k={}'.format(n + 1, RUN_TIMES, mr_info,
                                                                         res_labels[alabel], input_par_tr.n_per_label))

                x_train, y_train = dp.type_x_dataset('B', x_train_order, y_train_order, list_index_train, input_par_tr)

                time_start = time.time()

                utils.train_and_record(model_name, x_train, y_train, x_train, y_train,
                                       x_test, y_test,
                                       BATCH_SIZE, EPOCHS, LR_init,
                                       res_path, input_par_tr, input_par_te, is_record=flag_save_data)

                time_end = time.time()
                print('\nTime_len={}'.format(time_end-time_start))

        elif flag_train_mr == 'C':
            print('Type C Training set')

            input_par_tr.n_follow_up_samples = int(input_par_tr.percent * input_par_tr.n_per_label)*len(res_labels)

            res_path = './test_res/percent' + str(mr_per) + \
                       '/C_' + mr_info + '_k' + str(input_par_tr.n_per_label) + '/'
            utils.init_xlsx(res_path, input_par_tr, input_par_te, res_labels)

            for n in range(RUN_TIMES):

                print('\nRun times = {}/{}. MR={}. \nk={}'.format(n + 1, RUN_TIMES,
                                                                  mr_info, input_par_tr.n_per_label))

                x_train, y_train = dp.type_x_dataset('C', x_train_order, y_train_order, list_index_train, input_par_tr)

                time_start = time.time()

                utils.train_and_record(model_name, x_train, y_train, None, None, x_test, y_test,
                                       BATCH_SIZE, EPOCHS, LR_init,
                                       res_path, input_par_tr, input_par_te, is_record=flag_save_data)

                time_end = time.time()
                print('\nTime_len={}'.format(time_end - time_start))

        else:
            print('Not found Type {} Training set'.format(flag_train_mr))

def mnist(config_dict):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 指定0号GPU
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # 只显示error

    print('\nLoading MNIST')

    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.read_mnist("./MNIST_data/")  # 读数据

    IMG_SIZE = x_train.shape[1]
    IMG_CHAN = x_train.shape[-1]
    CLASS_NUM = y_train.shape[-1]
    print('img_size', IMG_SIZE)

    x_data = np.concatenate((x_train, x_test, x_valid), axis=0)
    y_data = np.concatenate((y_train, y_test, y_valid), axis=0)

    flag_order_load = True
    # data_order
    if os.path.exists('./order_data/x_order.npy') and flag_order_load:
        print('\nLoad Train order data')
        x_order_org = np.load('./order_data/x_order.npy')
        y_order_org = np.load('./order_data/y_order.npy')
        list_index_org = np.load('./order_data/list_index.npy')
    else:
        print('\nOrdering...')
        x_order_org, y_order_org, list_index_org = dp.order_data_set(x_data, y_data)
        os.makedirs('./order_data/', exist_ok=True)
        np.save('./order_data/x_order.npy', x_order_org)
        np.save('./order_data/y_order.npy', y_order_org)
        np.save('./order_data/list_index.npy', list_index_org)

    print(list_index_org)
    x_train_order_org, y_train_order_org, x_test_order_org, y_test_order_org \
        = dp.divide_train_test(x_order_org, y_order_org, list_index_org, n_per_tr=5000, tr_over_te=5)
    list_index_train_org = list(range(0, 5000*CLASS_NUM, 5000))
    list_index_test_org = list(range(0, 1000*CLASS_NUM, 1000))
    print(x_train_order_org.shape)
    print(list_index_train_org)

    flag_train_mr = config_dict['type_name']   # 'A', 'B', 'C'

    BATCH_SIZE = 100
    EPOCHS = config_dict['epochs']  # 50-k=2500, 150-k=5000
    LR_init = config_dict['lr']
    RUN_TIMES = config_dict['runtime']
    model_name = config_dict['model']
    k = config_dict['k']
    mr_per = config_dict['percent']
    mr_name = config_dict['mr_name']
    alabel = config_dict['alabel']
    flag_save_data = config_dict['save']

    drop_label = config_dict['drop_labels']
    res_label = list()
    for i in range(CLASS_NUM):
        if i in drop_label:
            res_label.append(i)
    res_label.sort()

    x_train_order, y_train_order, list_index_train = dp.sub_dataset(x_train_order_org, y_train_order_org,
                                                                    list_index_train_org, drop_labels)
    print('x_tr_order_sub shape = {0}. y_tr_order_sub shape = {1}'.format(x_train_order.shape, y_train_order.shape))
    print(list_index_train)

    x_test_order, y_test_order, list_index_test = dp.sub_dataset(x_test_order_org, y_test_order_org,
                                                                 list_index_test_org, drop_labels)

    print('x_te_order_sub shape = {0}. y_te_order_sub shape = {1}'.format(x_test_order.shape, y_test_order.shape))
    print(list_index_test)

    input_par_te = InputPar.InputPar('A', IMG_SIZE, IMG_CHAN, len(res_label),
                                     n_per_label=k // 5)
    x_test, y_test, _ = dp.type_x_dataset('A', x_test_order, y_test_order, list_index_test,
                                          input_par_te)

    if flag_train_mr == 'A':
        print('Type A Training set')

        input_par_tr = InputPar.InputPar(flag_train_mr, IMG_SIZE, IMG_CHAN, len(res_label),
                                         n_per_label=k)
        res_path = './' + model_name + '/test_res/' + str(res_label) + \
                   '/A_' + str(res_label) + '/A_k' + str(input_par_tr.n_per_label) + '/'
        utils.init_xlsx(res_path, input_par_tr, input_par_te, res_label)

        for n in range(RUN_TIMES):

            print('\nRun times = {}/{}'.format(n + 1, RUN_TIMES))
            time_start = time.time()

            x_train, y_train, _ = dp.type_x_dataset('A', x_train_order, y_train_order, list_index_train,
                                                    input_par_tr)
            print('\nx_train shape = {0}. y_train shape = {1}'.format(x_train.shape, y_train.shape))

            utils.train_and_record(model_name, x_train, y_train, None, None, x_test, y_test,
                                   BATCH_SIZE, EPOCHS, LR_init,
                                   res_path, input_par_tr, input_par_te, flag_save_data)

            time_end = time.time()

            print('\nTime_len={}'.format(time_end - time_start))

    else:

        input_par_tr = InputPar.InputPar(flag_train_mr, IMG_SIZE, IMG_CHAN, len(res_label),
                                         n_per_label=k,
                                         percent=1-mr_per)
        input_par_tr.mr_name = mr_name
        input_par_tr.mr_par = None
        mr_info = input_par_tr.mr_name + str(input_par_tr.mr_par)

        if flag_train_mr == 'B':
            print('Type B Training set')
            input_par_tr.n_follow_up_samples = int((1-input_par_tr.percent)*input_par_tr.n_per_label)

            input_par_tr.alabel = alabel

            res_path = './' + model_name + '/test_res/' + str(res_label) + '/percent'+str(mr_per) +\
                       '/B_' + mr_info + '_label' + str(res_label[alabel]) \
                       + '_k' + str(input_par_tr.n_per_label) + '/'
            utils.init_xlsx(res_path, input_par_tr, input_par_te, res_label)

            for n in range(RUN_TIMES):
                print('\nRun times = {}/{}. MR={}. \nAL={}. k={}'.format(n + 1, RUN_TIMES, mr_info,
                                                                         res_label[alabel],
                                                                         input_par_tr.n_per_label))

                x_train, y_train = dp.type_x_dataset('B', x_train_order, y_train_order, list_index_train, input_par_tr)

                time_start = time.time()

                utils.train_and_record(model_name, x_train, y_train, None, None, x_test, y_test,
                                       BATCH_SIZE, EPOCHS, LR_init,
                                       res_path, input_par_tr, input_par_te, flag_save_data)

                time_end = time.time()
                print('\nTime_len={}'.format(time_end-time_start))

        elif flag_train_mr == 'C':
            print('Type C Training set')

            input_par_tr.n_follow_up_samples = int(input_par_tr.percent * input_par_tr.n_per_label)*len(res_label)

            res_path = './' + model_name + '/test_res/' + str(res_label) + \
                       '/percent' + str(mr_per) + \
                       '/C_' + mr_info + '_k' + str(input_par_tr.n_per_label) + '/'
            utils.init_xlsx(res_path, input_par_tr, input_par_te, res_label)

            for n in range(RUN_TIMES):

                print('\nRun times = {}/{}. MR={}. \nk={}'.format(n + 1, RUN_TIMES,
                                                                  mr_info, input_par_tr.n_per_label))

                x_train, y_train = dp.type_x_dataset('C', x_train_order, y_train_order, list_index_train, input_par_tr)

                time_start = time.time()

                utils.train_and_record(model_name, x_train, y_train, None, None, x_test, y_test,
                                       BATCH_SIZE, EPOCHS, LR_init,
                                       res_path, input_par_tr, input_par_te, flag_save_data)

                time_end = time.time()
                print('\nTime_len={}'.format(time_end - time_start))

        else:
            print('Not found Type {} Training set'.format(flag_train_mr))

