import tkinter as tk
import tkinter.ttk as ttk
from tkinter import scrolledtext
import yaml

from ConnectHost import *
from RunClass import Run, RunInferace
from ConnectHost import RemoteHost


def read_yaml(filepath):

    return yaml.load(open(filepath, 'r', encoding="utf-8"))


class MainWindow:
    def __init__(self, rh):
        self.mr_set = read_yaml("./RunConfig/MRSet.yaml")
        self.dataset_and_model = read_yaml("./RunConfig/ModelandDataset.yaml")
        self.remote_host = rh

    def go_tr_model_com(self, *args):
        self.run_train.get_model(self.tr_model_var.get())
        print(self.tr_model_var.get())

    def go_tr_dataset_com(self, *args):
        self.run_train.get_dataset(self.tr_dataset_var.get())
        self.tr_mr_com["value"] = tuple(self.mr_set[self.tr_dataset_var.get()])
        print(self.tr_dataset_var.get())
        print(tuple(self.mr_set[self.tr_dataset_var.get()]))

    def go_tr_mr_com(self, *args):
        self.run_train.get_mr(self.tr_mr_var.get())
        print(self.tr_mr_var.get())
        # print(tr_mr_com.get())

    def go_tr_type_com(self, *args):
        self.run_train.get_type(self.tr_type_var.get())
        print(self.tr_type_var.get())
        # print(tr_mr_com.get())
        if self.tr_type_com.get() != 'B':
            self.tr_alabel_var.set(-1)
            self.tr_alabel_text.configure(state='disabled')
        else:
            self.tr_alabel_var.set(0)
            self.tr_alabel_text.configure(bg="white", state='normal')
        if self.tr_type_com.get() == 'A':
            self.tr_percent_var.set(0)
            self.tr_percent_text.config(state='disabled')
        else:
            self.tr_percent_text.config(bg="white", state='normal')

    def go_tr_save_checkbutton_com(self, *args):

        if self.tr_save_var.get() == 0:
            print(self.tr_save_var.get())
            self.tr_download_button.config(state="disabled")
        else:
            print(self.tr_save_var.get())
            self.tr_download_button.config(state="normal")

    def go_te_model_com(self, *args):
        self.run_test.get_model(self.te_model_com.get())
        self.te_dataset_com["value"] = self.dataset_and_model[self.te_model_com.get()]["dataset"]
        print(self.te_model_com.get())

    def go_te_dataset_com(self, *args):
        self.run_test.get_dataset(self.te_dataset_com.get())
        self.te_mr_com["value"] = tuple(self.mr_set[self.te_dataset_com.get()])
        print(self.te_dataset_com.get())

    def go_te_mr_com(self, *args):
        self.run_test.get_mr(self.te_mr_com.get())

    def go_te_save_checkbutton_com(self, *args):

        if self.te_save_var.get() == 0:
            print(self.te_save_var.get())
            self.te_download_button.config(state="disabled")
        else:
            print(self.te_save_var.get())
            self.te_download_button.config(state="normal")

    def Inferface(self):

        MainWin = tk.Tk()

        MainWin.title('Using MR to Train&Test DNN')

        MainWin.geometry('600x400')

        # self.remote_host = RemoteHost()
        self.menubar = tk.Menu(MainWin)  # 创建菜单栏
        self.menubar.add_command(label="Remote Host", command=self.remote_host.Interface)
        # menubar.add_command(label="Management", command=show_models)

        self.tab_right = ttk.Notebook()  # 创建分页栏
        self.tab_right.place(relx=0.52, rely=0.02, relwidth=0.46, relheight=0.96)
        self.tab_run = tk.Frame(self.tab_right)
        self.tab_run.pack()
        self.tab_right.add(self.tab_run, text="Run")

        self.stop_run_buttom = tk.Button(self.tab_right, text='Stop', command=self.remote_host.stop_process)
        self.stop_run_buttom.place(relx=0.8, rely=0.9)

        self.run_st = scrolledtext.ScrolledText(self.tab_run, wrap='word')
        self.run_st.pack()

        self.tab_left = ttk.Notebook()  # 创建分页栏
        self.tab_left.place(relx=0.02, rely=0.02, relwidth=0.48, relheight=0.96)

        self.tab_train = tk.Frame(self.tab_left)
        self.tab_train.pack()
        self.tab_left.add(self.tab_train, text="Training")

        self.run_train = Run('Train', self.run_st, self.remote_host)
        self.run_train_viewer = RunInferace(self.run_train, self.run_st)

        self.tr_config_label = tk.Label(self.tab_train, text="Train Config", width=10, height=1)
        self.tr_config_label.place(relx=0.02, rely=0.01)
        self.tr_config_frame = tk.Frame(self.tab_train, bd=1, relief='sunken')
        self.tr_config_frame.place(relx=0.02, rely=0.08, relwidth=0.96, relheight=0.7)

        self.tr_model_label = tk.Label(self.tr_config_frame, text="Model:", width=6, height=1, anchor='nw')
        self.tr_model_label.place(relx=0.02, rely=0.03)
        self.tr_model_var = tk.StringVar()
        self.tr_model_com = ttk.Combobox(self.tr_config_frame, textvariable=self.tr_model_var, width=11)
        self.tr_model_com.place(relx=0.22, rely=0.03)
        self.tr_model_com["value"] = ("lenet5", "vgg16")
        self.run_train.config_obj_box['model'] = self.tr_model_com

        # tr_model_com.set("lenet5")

        self.tr_model_com.bind("<<ComboboxSelected>>", self.go_tr_model_com)

        self.tr_dataset_label = tk.Label(self.tr_config_frame, text="DataSet:", width=10, height=1, anchor='nw')
        self.tr_dataset_label.place(relx=0.02, rely=0.18)
        self.tr_dataset_var = tk.StringVar()
        self.tr_dataset_com = ttk.Combobox(self.tr_config_frame, textvariable=self.tr_dataset_var, width=11)
        self.tr_dataset_com.place(relx=0.22, rely=0.18)
        self.tr_dataset_com["value"] = ('MNIST(0,1,8)', 'MNIST(0,1,3,8)', 'MNIST', 'CIFAR10')
        # tr_dataset_com.set('MNIST(0,1,8)')
        self.run_train.config_obj_box['dataset'] = self.tr_dataset_com

        self.tr_mr_label = tk.Label(self.tr_config_frame, text="MR:", width=6, height=1, anchor='nw')
        self.tr_mr_label.place(relx=0.6, rely=0.03)
        self.tr_mr_var = tk.StringVar()
        self.tr_mr_com = ttk.Combobox(self.tr_config_frame, textvariable=self.tr_mr_var, width=7)
        self.tr_mr_com.place(relx=0.71, rely=0.03)
        self.run_train.config_obj_box['mr'] = self.tr_mr_com

        self.tr_dataset_com.bind("<<ComboboxSelected>>", self.go_tr_dataset_com)

        self.tr_mr_com.bind("<<ComboboxSelected>>", self.go_tr_mr_com)

        self.tr_type_label = tk.Label(self.tr_config_frame, text="Type:", width=5, height=1, anchor='nw')
        self.tr_type_label.place(relx=0.6, rely=0.18)
        self.tr_type_var = tk.StringVar()
        self.tr_type_com = ttk.Combobox(self.tr_config_frame, textvariable=self.tr_type_var, width=5)
        self.tr_type_com.place(relx=0.74, rely=0.18)
        self.tr_type_com["value"] = ("A", "B", "C")
        self.run_train.config_obj_box['type_name'] = self.tr_type_com

        self.tr_epochs_label = tk.Label(self.tr_config_frame, text="Epochs:", width=6, height=1, anchor='nw')
        self.tr_epochs_label.place(relx=0.02, rely=0.33)
        self.tr_epochs_var = tk.StringVar()
        self.tr_epochs_text = tk.Entry(self.tr_config_frame, textvariable=self.tr_epochs_var, width=10)
        self.tr_epochs_text.place(relx=0.22, rely=0.33)
        self.run_train.config_obj['epochs'] = self.tr_epochs_var
        self.run_train.config_obj_box['epochs'] = self.tr_epochs_text

        self.tr_lr_label = tk.Label(self.tr_config_frame, text="Init LR:", width=6, height=1, anchor='nw')
        self.tr_lr_label.place(relx=0.02, rely=0.48)
        self.tr_lr_var = tk.StringVar()
        self.tr_lr_text = tk.Entry(self.tr_config_frame, textvariable=self.tr_lr_var, width=10)
        self.tr_lr_text.place(relx=0.22, rely=0.48)
        self.run_train.config_obj['lr'] = self.tr_lr_var
        self.run_train.config_obj_box['lr'] = self.tr_lr_text

        self.tr_alabel_label = tk.Label(self.tr_config_frame, text="Alabel:", width=6, height=1, anchor='nw')
        self.tr_alabel_label.place(relx=0.6, rely=0.33)
        self.tr_alabel_var = tk.StringVar()
        self.tr_alabel_text = tk.Entry(self.tr_config_frame, textvariable=self.tr_alabel_var, width=6)
        self.tr_alabel_text.place(relx=0.78, rely=0.34)
        self.run_train.config_obj['alabel'] = self.tr_alabel_var
        self.run_train.config_obj_box['alabel'] = self.tr_alabel_text

        self.tr_type_com.bind("<<ComboboxSelected>>", self.go_tr_type_com)

        self.tr_k_label = tk.Label(self.tr_config_frame, text="K:", width=6, height=1, anchor='nw')
        self.tr_k_label.place(relx=0.6, rely=0.48)
        self.tr_k_var = tk.StringVar()
        self.tr_k_text = tk.Entry(self.tr_config_frame, width=9, textvariable=self.tr_k_var)
        self.tr_k_text.place(relx=0.71, rely=0.49)
        self.run_train.config_obj['k'] = self.tr_k_var
        self.run_train.config_obj_box['k'] = self.tr_k_text

        self.tr_percent_label = tk.Label(self.tr_config_frame, text="Percent:", width=6, height=1, anchor='nw')
        self.tr_percent_label.place(relx=0.6, rely=0.63)
        self.tr_percent_var = tk.StringVar()
        self.tr_percent_text = tk.Entry(self.tr_config_frame, textvariable=self.tr_percent_var, width=5)
        self.tr_percent_text.place(relx=0.8, rely=0.64)
        self.run_train.config_obj['percent'] = self.tr_percent_var
        self.run_train.config_obj_box['percent'] = self.tr_percent_text

        self.tr_runtime_label = tk.Label(self.tr_config_frame, text="RunTimes:", width=9, height=1, anchor='nw')
        self.tr_runtime_label.place(relx=0.02, rely=0.63)
        self.tr_runtime_var = tk.StringVar()
        self.tr_runtime_text = tk.Entry(self.tr_config_frame, width=8, textvariable=self.tr_runtime_var)
        self.tr_runtime_text.place(relx=0.26, rely=0.64)
        self.run_train.config_obj['runtime'] = self.tr_runtime_var
        self.run_train.config_obj_box['runtime'] = self.tr_runtime_text

        self.tr_file_label = tk.Label(self.tr_config_frame, text="File: ", width=5, height=1, anchor='nw')
        self.tr_file_label.place(relx=0.02, rely=0.86)
        self.tr_file_name_var = tk.StringVar()
        self.tr_file_name_var.set(self.run_train.host_file_name)
        self.tr_file_text = tk.Entry(self.tr_config_frame, textvariable=self.tr_file_name_var, width=25)
        self.tr_file_text.place(relx=0.15, rely=0.87)
        self.run_train.config_obj['host_file_name'] = self.tr_file_name_var
        self.run_train.config_obj_box['host_file_name'] = self.tr_file_text
        # print("main viewer tr host file", self.run_train.host_file_name)

        self.train_button = tk.Button(self.tr_config_frame, text="OK", command=self.run_train_viewer.Interface)
        self.train_button.place(relx=0.85, rely=0.86)

        self.tr_res_label = tk.Label(self.tab_train, text="Train Result", width=10, height=1)
        self.tr_res_label.place(relx=0.02, rely=0.8)
        self.tr_res_frame = tk.Frame(self.tab_train, bd=1, relief='sunken')
        self.tr_res_frame.place(relx=0.02, rely=0.86, relwidth=0.96, relheight=0.2)

        self.tr_prediction_label = tk.Label(self.tr_res_frame, text="Prediction Accuracy:",
                                            width=20, height=1, anchor='nw')
        self.tr_prediction_label.place(relx=0.02, rely=0.02)
        self.tr_prediction_var = tk.StringVar()
        self.tr_prediction_text = tk.Entry(self.tr_res_frame, textvariable=self.tr_prediction_var,
                                           width=15, bg="white", state="disable")
        self.tr_prediction_text.place(relx=0.02, rely=0.38)
        self.tr_prediction_var.set(self.run_train.res)
        self.run_train.config_obj['accuracy'] = self.tr_prediction_var

        self.tr_download_button = tk.Button(self.tr_res_frame, text="Download\ndata",
                                            command=self.run_train.download_data)
        self.tr_download_button.place(relx=0.7, rely=0.02)
        self.tr_download_button.config(state="disabled")
        self.run_train.config_obj_box['download'] = self.tr_download_button

        self.tr_save_label = tk.Label(self.tr_config_frame, text="Save res in .xlsx", width=15, height=1, anchor='nw')
        self.tr_save_label.place(relx=0.02, rely=0.75)
        self.tr_save_var = tk.IntVar()
        self.tr_save_var.set(0)
        self.tr_save_checkbutton = tk.Checkbutton(self.tr_config_frame, variable=self.tr_save_var)
        self.tr_save_checkbutton.place(relx=0.4, rely=0.74)
        self.run_train.config_obj['save'] = self.tr_save_var
        self.run_train.config_obj_box['save'] = self.tr_save_checkbutton

        self.tab_test = tk.Frame(self.tab_left)
        self.tab_test.pack()
        self.tab_left.add(self.tab_test, text="Testing")

        self.run_test = Run('Test', self.run_st, self.remote_host)
        self.run_test_viewer = RunInferace(self.run_test, self.run_st)

        self.te_config_label = tk.Label(self.tab_test, text="Test Config", width=10, height=1)
        self.te_config_label.place(relx=0.02, rely=0.01)
        self.te_config_frame = tk.Frame(self.tab_test, bd=1, relief='sunken')
        self.te_config_frame.place(relx=0.02, rely=0.08, relwidth=0.96, relheight=0.35)

        self.te_model_label = tk.Label(self.te_config_frame, text="Model:", width=8, height=1, anchor='nw')
        self.te_model_label.place(relx=0.02, rely=0.03)
        self.te_model_var = tk.StringVar()
        self.te_model_com = ttk.Combobox(self.te_config_frame, textvariable=self.te_model_var, width=10)
        self.te_model_com.place(relx=0.22, rely=0.03)
        self.te_model_com["value"] = ("lenet5", "vgg16")
        self.run_test.config_obj_box['model'] = self.te_model_com

        self.te_dataset_label = tk.Label(self.te_config_frame, text="DataSet:", width=8, height=1, anchor='nw')
        self.te_dataset_label.place(relx=0.02, rely=0.23)
        self.te_dataset_var = tk.StringVar()
        self.te_dataset_com = ttk.Combobox(self.te_config_frame, textvariable=self.te_dataset_var, width=10)
        self.te_dataset_com.place(relx=0.22, rely=0.23)
        # self.te_dataset_com["value"] = ("MNIST", "CIFAR10")
        self.run_test.config_obj_box['dataset'] = self.te_dataset_com

        self.te_mr_label = tk.Label(self.te_config_frame, text="MR:", width=8, height=1, anchor='nw')
        self.te_mr_label.place(relx=0.6, rely=0.03)
        self.te_mr_var = tk.StringVar()
        self.te_mr_com = ttk.Combobox(self.te_config_frame, textvariable=self.te_mr_var, width=7)
        self.te_mr_com.place(relx=0.71, rely=0.03)
        self.run_test.config_obj_box['mr'] = self.te_mr_com

        self.te_model_com.bind("<<ComboboxSelected>>", self.go_te_model_com)

        self.te_dataset_com.bind("<<ComboboxSelected>>", self.go_te_dataset_com)

        self.te_mr_com.bind("<<ComboboxSelected>>", self.go_te_mr_com)

        self.te_save_label = tk.Label(self.te_config_frame, text="Save res in .xlsx", width=15, height=1, anchor='nw')
        self.te_save_label.place(relx=0.54, rely=0.58)
        self.te_save_var = tk.IntVar()
        self.te_save_var.set(0)
        self.te_save_checkbutton = tk.Checkbutton(self.te_config_frame, variable=self.te_save_var)
        self.te_save_checkbutton.place(relx=0.9, rely=0.57)
        self.run_test.config_obj['save'] = self.te_save_var
        self.run_test.config_obj_box['save'] = self.te_save_checkbutton

        self.te_ksec_label = tk.Label(self.te_config_frame, text="k sec:", width=5, height=1, anchor='nw')
        self.te_ksec_label.place(relx=0.02, rely=0.44)
        self.te_ksec_var = tk.StringVar()
        self.te_ksec_text = tk.Entry(self.te_config_frame, textvariable=self.te_ksec_var, width=9)
        self.te_ksec_var.set(1000)
        self.te_ksec_text.place(relx=0.22, rely=0.44)
        self.run_test.config_obj['k_sec'] = self.te_ksec_var
        self.run_test.config_obj_box['k_sec'] = self.te_ksec_text

        self.te_topk_label = tk.Label(self.te_config_frame, text="top k:", width=5, height=1, anchor='nw')
        self.te_topk_label.place(relx=0.02, rely=0.62)
        self.te_topk_var = tk.StringVar()
        self.te_topk_text = tk.Entry(self.te_config_frame, textvariable=self.te_topk_var, width=9)
        self.te_topk_var.set(1)
        self.te_topk_text.place(relx=0.22, rely=0.62)
        self.run_test.config_obj['top_k'] = self.te_topk_var
        self.run_test.config_obj_box['top_k'] = self.te_topk_text

        self.te_k_label = tk.Label(self.te_config_frame, text="K:", width=8, height=1, anchor='nw')
        self.te_k_label.place(relx=0.6, rely=0.23)
        self.te_k_var = tk.StringVar()
        self.te_k_text = tk.Entry(self.te_config_frame, width=9, textvariable=self.te_k_var)
        self.te_k_text.place(relx=0.71, rely=0.24)
        self.run_test.config_obj['k'] = self.te_k_var
        self.run_test.config_obj_box['k'] = self.te_k_text

        self.te_runtime_label = tk.Label(self.te_config_frame, text="RunTimes:", width=10, height=1, anchor='nw')
        self.te_runtime_label.place(relx=0.6, rely=0.41)
        self.te_runtime_var = tk.StringVar()
        self.te_runtime_text = tk.Entry(self.te_config_frame, width=4, textvariable=self.te_runtime_var)
        self.te_runtime_text.place(relx=0.85, rely=0.42)
        self.run_test.config_obj['runtime'] = self.te_runtime_var
        self.run_test.config_obj_box['runtime'] = self.te_runtime_text

        self.te_file_label = tk.Label(self.te_config_frame, text="File: ", width=5, height=1, anchor='nw')
        self.te_file_label.place(relx=0.02, rely=0.8)
        self.te_file_name_var = tk.StringVar()
        self.te_file_name_var.set(self.run_test.host_file_name)
        self.te_file_text = tk.Entry(self.te_config_frame, textvariable=self.te_file_name_var, width=25)
        self.te_file_text.place(relx=0.14, rely=0.8)
        self.run_test.config_obj['host_file_name'] = self.te_file_name_var
        self.run_test.config_obj_box['host_file_name'] = self.te_file_text
        # print("main viewer te file name: ", self.run_test.host_file_name)

        self.test_button = tk.Button(self.te_config_frame, text="OK", command=self.run_test_viewer.Interface)
        self.test_button.place(relx=0.86, rely=0.74)

        self.te_res_label = tk.Label(self.tab_test, text="Test Result", width=10, height=1)
        self.te_res_label.place(relx=0.02, rely=0.44)
        self.te_res_frame = tk.Frame(self.tab_test, bd=1, relief='sunken')
        self.te_res_frame.place(relx=0.02, rely=0.5, relwidth=0.96, relheight=0.48)

        self.te_prediction_label = tk.Label(self.te_res_frame, text="Prediction Accuracy:",
                                            width=20, height=1, anchor='nw', justify='left')
        self.te_prediction_label.place(relx=0.02, rely=0.02)
        self.te_prediction_var = tk.StringVar()
        self.te_prediction_text = tk.Entry(self.te_res_frame, textvariable=self.te_prediction_var,
                                           width=15, bg="white", state="disable")
        self.te_prediction_text.place(relx=0.5, rely=0.03)
        self.te_prediction_var.set(self.run_test.res)
        self.run_test.config_obj['accuracy'] = self.te_prediction_var

        self.kmncov_label = tk.Label(self.te_res_frame, text="KMNCov:", width=10, height=1, anchor='nw')
        self.kmncov_label.place(relx=0.02, rely=0.22)
        self.kmncov_var = tk.StringVar()
        self.kmncov_text = tk.Entry(self.te_res_frame, textvariable=self.kmncov_var, width=15, state="disable")
        self.kmncov_text.place(relx=0.25, rely=0.22)
        self.kmncov_var.set(self.run_test.kmncov)
        self.run_test.config_obj['kmncov'] = self.kmncov_var

        self.nbcov_label = tk.Label(self.te_res_frame, text="NBCov:", width=10, height=1, anchor='nw')
        self.nbcov_label.place(relx=0.02, rely=0.37)
        self.nbcov_var = tk.StringVar()
        self.nbcov_text = tk.Entry(self.te_res_frame, textvariable=self.nbcov_var, width=15, state="disable")
        self.nbcov_text.place(relx=0.25, rely=0.37)
        self.nbcov_var.set(self.run_test.nbcov)
        self.run_test.config_obj['nbcov'] = self.nbcov_var

        self.sancov_label = tk.Label(self.te_res_frame, text="SANCov:", width=10, height=1, anchor='nw')
        self.sancov_label.place(relx=0.02, rely=0.52)
        self.sancov_var = tk.StringVar()
        self.sancov_text = tk.Entry(self.te_res_frame, textvariable=self.sancov_var, width=15, state="disable")
        self.sancov_text.place(relx=0.25, rely=0.52)
        self.sancov_var.set(self.run_test.sancov)
        self.run_test.config_obj['sancov'] = self.sancov_var

        self.tkncov_label = tk.Label(self.te_res_frame, text="TKNCov:", width=10, height=1, anchor='nw')
        self.tkncov_label.place(relx=0.02, rely=0.67)
        self.tkncov_var = tk.StringVar()
        self.tkncov_text = tk.Entry(self.te_res_frame, textvariable=self.tkncov_var, width=15, state="disable")
        self.tkncov_text.place(relx=0.25, rely=0.67)
        self.tkncov_var.set(self.run_test.tkncov)
        self.run_test.config_obj['tkncov'] = self.tkncov_var

        self.tknpat_label = tk.Label(self.te_res_frame, text="TKNPat:", width=10, height=1, anchor='nw')
        self.tknpat_label.place(relx=0.02, rely=0.82)
        self.tknpat_var = tk.StringVar()
        self.tknpat_text = tk.Entry(self.te_res_frame, textvariable=self.tknpat_var, width=15, state="disable")
        self.tknpat_text.place(relx=0.25, rely=0.82)
        self.tknpat_var.set(self.run_test.tknpat)
        self.run_test.config_obj['tknpat'] = self.tknpat_var

        self.te_download_button = tk.Button(self.te_res_frame, text="Download\ndata",
                                            command=self.run_test.download_data)
        self.te_download_button.place(relx=0.7, rely=0.68)
        self.te_download_button.config(state="disabled")
        self.run_test.config_obj_box['download'] = self.te_download_button

        # tab_mr = tk.Frame(tab_left, bg='white', bd=1)
        # tab_mr.pack()
        # tab_left.add(tab_mr, text='MG')
        # mr_label = tk.Label(tab_mr, text="Metamorphic Group", bg='white', width=30, height=1)
        # mr_label.pack()

        MainWin.config(menu=self.menubar)  # 显示菜单栏
        MainWin.mainloop()


def run_mainwin(rh):
    mainwin = MainWindow(rh)
    mainwin.Inferface()


if __name__ == "__main__":
    # with open("./RunConfig/TrainConfig.jason", "r", encoding='utf-8') as f:
    #     config_dict = json.load(f)
    # print(config_dict)
    # print(read_yaml("./RunConfig/MRSet.yaml"))
    rm = RemoteHost()
    mainwin = MainWindow(rm)
    mainwin.Inferface()