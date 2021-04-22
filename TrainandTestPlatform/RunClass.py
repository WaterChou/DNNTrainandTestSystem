import tkinter as tk
from SaveJason import SaveJson
import ConnectHost
import threading
import os
import yaml

class Run(threading.Thread):

    def __init__(self, name, viewer, remote_host):
        threading.Thread.__init__(self, name="RunInfo" + name)
        self.event = threading.Event()

        self.name = name
        self.viewer = viewer
        self.remote_host = remote_host

        self.config_obj = dict()
        self.config_obj_box = dict()

        self.percent = 0.5
        self.epochs = 0
        self.lr = 0
        self.alabel = -1
        self.runtime = 0
        self.k = 0
        self.k_sec = 0
        self.top_k = 0

        self.model = 'lenet5'
        self.dataset = 'MNIST(0,1,8)'
        self.mr_name = "mr_name"
        self.type_name = "type_name"

        self.save_flag = False
        self.drop_label = []

        self.host_file_name = "{}MR{}/".format(remote_host.host_filename, name)
        self.local_config_path = "./RunConfig/{}Config.jason".format(name)  # local
        self.host_config_path = "{}RunConfig/{}Config.jason".format(self.host_file_name, name)  # local

        self.kmncov = 0
        self.nbcov = 0
        self.sancov = 0
        self.tkncov = 0
        self.tknpat = 0

        self.res = 0
        self.stop = False
        self.end = False

    def get_model(self, model):
        self.model = model

    def get_dataset(self, dataset):
        self.dataset = dataset

    def get_mr(self, mr_name):
        self.mr_name = mr_name

    def get_type(self, type_name):
        self.type_name = type_name

    def set_config(self):
        print("set config")

        self.config_obj['accuracy'].set(0)
        # self.stop = False

        if self.name == 'Train':
            self.epochs = int(self.config_obj['epochs'].get())
            self.lr = float(self.config_obj['lr'].get())
            self.percent = float(self.config_obj['percent'].get())
            if self.type_name == "B":
                self.alabel = int(self.config_obj['alabel'].get())
        else:

            init_config = yaml.load(open("./RunConfig/ModelandDataset.yaml", 'r', encoding="utf-8"))
            self.epochs = init_config[self.model]['epochs']
            self.lr = init_config[self.model]['lr']
            self.k_sec = int(self.config_obj['k_sec'].get())
            self.top_k = int(self.config_obj['top_k'].get())

        self.runtime = int(self.config_obj['runtime'].get())
        self.k = int(self.config_obj['k'].get())
        self.host_file_name = self.config_obj['host_file_name'].get()
        self.save_flag = bool(self.config_obj['save'].get())

        for k in self.config_obj_box.keys():
            self.config_obj_box[k].config(state="disabled")

        # self.remote_host.connect()
        print(self.name, "set config host file name", self.host_file_name)
        self.remote_host.local_filename = "./RunConfig/{}Config.jason".format(self.name)
        self.remote_host.host_filename = "{}RunConfig/{}Config.jason".format(self.host_file_name,
                                                                             self.name)
        self.remote_host.run_config = self
        # self.remote_host.run_interface = self.viewer

        item = {'name': self.name,
                'model': self.model,
                'dataset': self.dataset,
                'type_name': self.type_name,
                'alabel': self.alabel,
                'mr_name': self.mr_name,
                'k': self.k,
                'drop_label': self.drop_label,
                'percent': self.percent,
                'epochs': self.epochs,
                'lr': self.lr,
                'k_sec': self.k_sec,
                'top_k': self.top_k,
                'runtime': self.runtime,
                'save': self.save_flag,
                'file_name': self.host_file_name}

        SaveJson().save_file("./RunConfig/", "{}Config.jason".format(self.name), item)
        self.remote_host.put(self.local_config_path, self.host_config_path)

    def get_res(self):
        print("get res")
        for k in self.config_obj_box.keys():
            if k != "download":
                self.config_obj_box[k].config(state="normal")
            elif self.save_flag:
                self.config_obj_box[k].config(state="normal")

        self.remote_host.download(self.host_config_path,
                                  self.local_config_path)
        res_dicts = SaveJson().read_file(self.local_config_path)
        self.res = res_dicts['accuracy']

        if self.name == "Test":
            self.kmncov = res_dicts['kmncov']
            self.nbcov = res_dicts['nbcov']
            self.sancov = res_dicts['sancov']
            self.tkncov = res_dicts['tkncov']
            self.tknpat = res_dicts['tknpat']

    def download_data(self):

        if self.name == 'Train':
            if self.type_name == 'A':
                remotepath = "{}test_res/{}/A_k{}/acc_label.xlsx".format(self.host_file_name,
                                                                         self.dataset, self.k)
                localpath = "./test_res/{}/A_k{}/".format(self.dataset, self.k)
                # os.makedirs(localpath, exist_ok=True)
                # localpath += "acc_label.xlsx"

            elif self.type_name == 'B':
                remotepath = "{}test_res/{}/percent{}/" \
                             "B_{}None_label{}_k{}/acc_label.xlsx".format(self.host_file_name,
                                                                          self.dataset, self.percent,
                                                                          self.mr_name, self.alabel, self.k)
                localpath = "./test_res/{}/percent{}/" \
                             "B_{}None_label{}_k{}/".format(self.dataset, self.percent,
                                                            self.mr_name, self.alabel, self.k)
            else:
                remotepath = "{}test_res/{}/percent{}/" \
                             "C_{}None_k{}/acc_label.xlsx".format(self.host_file_name,
                                                                  self.dataset, self.percent,
                                                                  self.mr_name, self.k)
                localpath = "./test_res/{}/percent{}/" \
                             "C_{}None_k{}/".format(self.dataset, self.percent,
                                                    self.mr_name, self.k)
            os.makedirs(localpath, exist_ok=True)
            localpath += "acc_label.xlsx"
        else:
            remotepath = "{}cov_res/{}/{}/k{}/neuron_cov.xlsx".format(self.host_file_name,
                                                                      self.dataset, self.mr_name, self.k)
            localpath = "./cov_res/{}/{}/k{}/".format(self.dataset, self.mr_name, self.k)
            os.makedirs(localpath, exist_ok=True)
            localpath += "neuron_cov.xlsx"

        print("\nDownload from: ", remotepath)
        self.remote_host.download(remotepath, localpath)
        self.viewer.insert('insert', "Download to: \n{}\n".format(localpath))
        self.viewer.yview('end')


class RunInferace:

    def __init__(self, run_config, viewer):
        self.viewer = viewer
        self.run_config = run_config

    def update_res(self):
        print('stop flag', self.run_config.stop)
        self.run_config.event.clear()
        self.run_config.event.wait()
        if self.run_config.stop is False:
            if self.run_config.save_flag is True:
                self.run_config.config_obj_box['download'].config(state='normal')
            self.run_config.get_res()
            self.run_config.config_obj['accuracy'].set(round(self.run_config.res, 5))
            if self.run_config.name == "Test":
                self.run_config.config_obj['kmncov'].set(round(self.run_config.kmncov, 5))
                self.run_config.config_obj['nbcov'].set(round(self.run_config.nbcov, 5))
                self.run_config.config_obj['sancov'].set(round(self.run_config.sancov, 5))
                self.run_config.config_obj['tkncov'].set(round(self.run_config.tkncov, 5))
                self.run_config.config_obj['tknpat'].set(round(self.run_config.tknpat, 5))
        self.run_config.event.clear()

    def Interface(self):

        print('\nrun interface')
        self.viewer.delete(1.0, 'end')
        self.viewer.insert('insert', "Remote {}ing...\n".format(self.run_config.name))

        self.run_config.stop = False

        self.run_config.set_config()

        t_get_res = threading.Thread(target=self.update_res)
        t_get_res.start()

        cm = "anaconda3/envs/tf/bin/python {}MR{}.py".format(self.run_config.host_file_name, self.run_config.name)
        t_rm = threading.Thread(target=self.run_config.remote_host.exec_command, args=(cm,))
        t_rm.start()



