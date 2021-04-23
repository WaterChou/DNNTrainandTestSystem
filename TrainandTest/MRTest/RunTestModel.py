import os
import numpy as np
import utils
import dataprocess as dp
import tensorflow as tf
import time
from myclass import InputPar


class RunTestModel:
    def __init__(self, config_dict):
        self.dataset = config_dict['dataset']
        self.model = config_dict['model']
        self.k = config_dict['k']
        self.mr_name = config_dict['mr_name']
        self.RUN_TIMES = config_dict['runtime']
        self.EPOCHS = config_dict['epochs']
        self.LR_init = config_dict['lr']
        self.flag_save_data = config_dict['save']

        self.k_sec = config_dict['k_sec']
        self.top_k = config_dict['top_k']

        self.file_path = config_dict['file_name']

        self.acc = 0
        self.kmncov = 0
        self.nbcov = 0
        self.sancov = 0
        self.tkncov = 0
        self.tknpat = 0

    def get_mnist_data(self):

        print('\nLoading MNIST')
        order_data_path = "{}order_data/MNIST/".format(self.file_path)
        x_train, y_train, x_valid, y_valid, x_test, y_test = utils.read_mnist("{}MNIST_data/".format(order_data_path))  # 读数据

        self.IMG_SIZE = x_train.shape[1]
        self.IMG_CHAN = x_train.shape[-1]
        self.CLASS_NUM = y_train.shape[-1]
        self.res_label = list(range(self.CLASS_NUM))
        print('img_size', self.IMG_SIZE)
        print(self.res_label)

        x_data = np.concatenate((x_train, x_test, x_valid), axis=0)
        y_data = np.concatenate((y_train, y_test, y_valid), axis=0)

        # data_order
        print(order_data_path)
        print(os.path.exists('{}x_order.npy'.format(order_data_path)))

        if os.path.exists('{}x_order.npy'.format(order_data_path)):
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

        print(list_index_org)

        x_train_order_org, y_train_order_org, x_test_order_org, y_test_order_org \
            = dp.divide_train_test(x_order_org, y_order_org, list_index_org, n_per_tr=5000, tr_over_te=5)

        list_index_train_org = list(range(0, 5000 * self.CLASS_NUM, 5000))
        list_index_test_org = list(range(0, 1000 * self.CLASS_NUM, 1000))
        print(x_train_order_org.shape)
        print(list_index_train_org)

        self.x_train_order = x_train_order_org
        self.y_train_order = y_train_order_org
        self.list_index_train = list_index_train_org

        print('x_tr_order_sub shape = {0}. y_tr_order_sub shape = {1}'.format(self.x_train_order.shape,
                                                                              self.y_train_order.shape))
        print(self.list_index_train)

        self.x_test_order = x_test_order_org
        self.y_test_order = y_test_order_org
        self.list_index_test = list_index_test_org

        print('x_te_order_sub shape = {0}. y_te_order_sub shape = {1}'.format(self.x_test_order.shape,
                                                                              self.y_test_order.shape))
        print(self.list_index_test)

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
        self.res_label = list(range(self.CLASS_NUM))

        y_train = utils.label_dense_to_one_hot(y_train, self.CLASS_NUM)
        y_test = utils.label_dense_to_one_hot(y_test, self.CLASS_NUM)

        # train_order
        order_data_path = "{}order_data/CIFAR10/".format(self.file_path)
        print(order_data_path)
        print(os.path.isfile('{}x_train_order.npy'.format(order_data_path)))

        if os.path.isfile('{}x_train_order.npy'.format(order_data_path)):
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

        print('x_tr_order shape = {0}. y_tr_order shape = {1}'.format(x_train_order.shape, y_train_order.shape))
        print(list_index_train)

        self.x_train_order = x_train_order
        self.y_train_order = y_train_order
        self.list_index_train = list_index_train

        # test_order
        if os.path.exists('{}x_test_order.npy'.format(order_data_path)):
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


        self.x_test_order = x_test_order
        self.y_test_order = y_test_order
        self.list_index_test = list_index_test

        print('x_te_order shape = {0}. y_te_order shape = {1}'.format(self.x_test_order.shape, self.y_test_order.shape))
        print(self.list_index_test)

    def run(self):

        if self.dataset == 'CIFAR10':
            self.get_cifar10_data()
        else:
            self.get_mnist_data()

        input_par_tr = InputPar.InputPar('A', self.IMG_SIZE, self.IMG_CHAN, len(self.res_label),
                                         n_per_label=5000)
        x_train, y_train, _ = dp.type_x_dataset('A', self.x_train_order, self.y_train_order,
                                                self.list_index_train,
                                                input_par_tr)

        sess, env = utils.train(input_par_tr, self.model,
                                x_train, y_train, x_train, y_train, epochs=self.EPOCHS, lr_init=self.LR_init, batch_size=128)

        print('\nget dict mfr')
        time_start = time.time()

        dict_path = "{}dict/{}/".format(self.file_path, self.model)
        dict_mfr = utils.get_dict_mfr_data(sess, env, x_train, y_train, batch_size=128, file_path=dict_path,
                                           load_dict_mfr_flag=True)
        print("time len = {}".format(time.time() - time_start))

        input_par_te = InputPar.InputPar('C', self.IMG_SIZE, self.IMG_CHAN, len(self.res_label), percent=0.5,
                                         mr_name=self.mr_name, n_per_label=self.k)

        for run_i in range(self.RUN_TIMES):
            print('\nRun times = {}/{}'.format(run_i + 1, self.RUN_TIMES))
            x_test, y_test = dp.type_x_dataset('C', self.x_test_order, self.y_test_order,
                                                  self.list_index_test, input_par_te)
            print("x shape", x_test.shape)
            loss, self.acc = utils.evaluate(sess, env, x_test, y_test, batch_size=128)
            self.kmncov, self.nbcov, self.sancov, _ = utils.neuron_level_cov(sess, env, x_test, y_test, dict_mfr,
                                                                             self.k_sec)
            self.tkncov, self.tknpat = utils.layer_level_cov(sess, env, x_test, y_test, self.top_k, batch_size=128)

            if self.flag_save_data:
                file_path = '{}cov_res/{}/{}/k{}/'.format(self.file_path, self.dataset, self.mr_name, self.k)
                # file_path = './MRTest/cov_res/{}/{}/k{}/'.format(self.dataset, self.mr_name, self.k)
                print(file_path)
                os.makedirs(file_path, exist_ok=True)
                utils.update_xlsx(file_path + 'neuron_cov.xlsx',
                                  [input_par_te.percent, self.acc,
                                   self.kmncov, self.nbcov, self.sancov, self.tkncov, self.tknpat])

            print("acc={}".format(self.acc))
            print("kmncov={}".format(self.kmncov))
            print("nbcov={}".format(self.nbcov))
            print("sancov={}".format(self.sancov))
            print("tkncov = {}".format(self.tkncov))
            print("tkpnat = {}".format(self.tknpat))
            print("time len = {}".format(time.time() - time_start))

        sess.close()
