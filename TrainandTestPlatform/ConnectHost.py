import tkinter as tk
import paramiko
import json
import threading
import yaml


class RemoteHost(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name="Remotehost")
        self.event = None

        self.config_dict = dict()
        self.client = None
        self.hostip = "192.168.3.52"
        self.port = 22
        self.username = "xiaohui"
        self.password = "123456"

        self.selected_host_index = 0

        self._transport = None
        self._sftp = None

        self.host_filename = "PycharmProjects/Chou_Jnyi/Share/TrainandTest/"
        self.local_filename = None

        self.run_config = None

        self.hostlist_dict = yaml.load(open("./RunConfig/HostList.yaml", 'r', encoding="utf-8"))

    def get_config(self):
        self.hostip = self.config_dict['hostip'].get()
        self.port = int(self.config_dict['port'].get())
        self.username = self.config_dict['user'].get()
        self.password = self.config_dict['pw'].get()

        self.connect()

        print(self.hostip)
        print(self.port)
        print(self.username)
        print(self.password)

        self.go_save_host_info()

        self.viewer.destroy()

    def connect(self):
        transport = paramiko.Transport((self.hostip, self.port))
        transport.connect(username=self.username, password=self.password)

        self._transport = transport

    def download(self, remotepath, localpath):  # download
        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        self._sftp.get(remotepath, localpath)

    def put(self, localpath, remotepath):   # upload

        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        self._sftp.put(localpath, remotepath)

    def exec_command(self, command):
        print("exec_command")
        if self.client is None:
            self.client = paramiko.SSHClient()
            self.client._transport = self._transport
        stdin, stdout, stderr = self.client.exec_command(command, get_pty=True)

        while not stdout.channel.exit_status_ready():
            result = stdout.readline()
            # 由于在退出时，stdout还是会有一次输出，因此需要单独处理，处理完之后，就可以跳出了
            self.run_config.viewer.insert('insert', result)
            self.run_config.viewer.yview('end')
            if stdout.channel.exit_status_ready():
                # a = stdout.readlines()
                # print(a)
                break
        self.run_config.event.set()
        print("exec_command end")
    # def exec_command(self, command, outer_viewer=None):
    #     if self.client is None:
    #         self.client = paramiko.SSHClient()
    #         self.client._transport = self._transport
    #     stdin, stdout, stderr = self.client.exec_command(command, get_pty=True)
    #     while not stdout.channel.exit_status_ready():
    #         result = stdout.readline()
    #         print(result)
    #         # yield result
    #         if outer_viewer is not None:
    #             outer_viewer.insert('insert', result)
    #             outer_viewer.update()
    #             outer_viewer.yview('end')
    #
    #         # 由于在退出时，stdout还是会有一次输出，因此需要单独处理，处理完之后，就可以跳出了
    #         if stdout.channel.exit_status_ready():
    #             a = stdout.readlines()
    #             print(a)
    #             break
        # data = stdout.read()
        # if len(data) > 0:
        #     print
        #     data.strip()  # 打印正确结果
        #     return data
        # err = stderr.read()
        # if len(err) > 0:
        #     print
        #     err.strip()  # 输出错误结果
        #     return err

    def close(self):
        if self._transport:
            self._transport.close()
        if self.client:
            self.client.close()

    def stop_process(self):
        print("stop process")
        print(self.host_filename)
        self.download(self.host_filename, self.local_filename)
        run_config = json.load(open(self.local_filename))
        pid = run_config['pid']
        self.run_config.stop = True
        cm = "kill {}".format(pid)
        self.exec_command(cm)

        self.run_config.viewer.insert('insert', "\n"+"="*15+"Kill"+"="*15)
        self.run_config.viewer.yview('end')
        # self.run_config.event.wait()
        for k in self.run_config.config_obj_box.keys():
            if k != "download":
                self.run_config.config_obj_box[k].config(state="normal")

        print("stop process end")

    def go_save_host_info(self):

        print("save host info", self.save_host_info_var.get())
        if self.save_host_info_var.get():

            new_host_info_dict = {
                "hostip": self.config_dict['hostip'].get(),
                "port":  int(self.config_dict['port'].get()),
                "user": self.config_dict['user'].get(),
                "pw": self.config_dict['pw'].get(),
                "filename": self.config_dict['filename'].get()
            }

            flag_exist = False
            for item in self.hostlist_dict:
                if self.hostlist_dict[item] == new_host_info_dict:
                    flag_exist = True
                    break
            if flag_exist is False:
                print(self.hostlist_dict.keys())
                tmp_list = list(self.hostlist_dict.keys())
                tmp_list.sort()
                print(tmp_list)
                host_num = int(tmp_list[-1][5:]) + 1

                print(host_num)
                self.hostlist_dict["host_"+str(host_num+1)] = new_host_info_dict
                # 写入到yaml文件
                with open("./RunConfig/HostList.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(self.hostlist_dict, f)

    def update_interface(self):

        host_index = self.hostlist[self.config_dict["hostlist"].curselection()[0]]
        self.hostip = self.hostlist_dict[host_index]["hostip"]
        self.username = self.hostlist_dict[host_index]["user"]
        self.port = self.hostlist_dict[host_index]["port"]
        self.password = self.hostlist_dict[host_index]["pw"]
        self.host_filename = self.hostlist_dict[host_index]["filename"]

        self.config_dict["user"].set(self.username)
        self.config_dict["hostip"].set(self.hostip)
        self.config_dict["port"].set(self.port)
        self.config_dict["pw"].set(self.password)

        self.hostlist_viewer.destroy()

    def go_delete_host(self):

        host_index = self.config_dict["hostlist"].curselection()[0]

        del(self.hostlist_dict[self.hostlist[host_index]])
        self.config_dict["hostlist"].delete(host_index)
        del(self.hostlist[host_index])
        # 写入到yaml文件
        with open("./RunConfig/HostList.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.hostlist_dict, f)


    def Interface(self):
        print('connect host')
        RemoteWin = tk.Toplevel()
        RemoteWin.title("SSH Remote Host")
        RemoteWin.geometry('350x200')

        self.viewer = RemoteWin

        ip_label = tk.Label(RemoteWin, text="Host IP: ", width=10)
        ip_label.place(relx=0.02, rely=0.02)
        hostip_var = tk.StringVar()
        hostip_var.set(self.hostip)
        ip_text = tk.Entry(RemoteWin, textvariable=hostip_var, width=20)
        ip_text.place(relx=0.3, rely=0.02)
        self.config_dict['hostip'] = hostip_var

        port_label = tk.Label(RemoteWin, text="Port:", width=10)
        port_label.place(relx=0.02, rely=0.19)
        port_var = tk.StringVar()
        port_var.set(self.port)
        port_text = tk.Entry(RemoteWin, textvariable=port_var, width=20)
        port_text.place(relx=0.3, rely=0.19)
        self.config_dict['port'] = port_var

        user_label = tk.Label(RemoteWin, text="UserName:", width=10)
        user_label.place(relx=0.02, rely=0.36)
        user_var = tk.StringVar()
        user_var.set(self.username)
        user_text = tk.Entry(RemoteWin, textvariable=user_var, width=20)
        user_text.place(relx=0.3, rely=0.36)
        self.config_dict['user'] = user_var

        pw_label = tk.Label(RemoteWin, text="Password:", width=10)
        pw_label.place(relx=0.02, rely=0.53)
        pw_var = tk.StringVar()
        pw_var.set(self.password)
        pw_text = tk.Entry(RemoteWin, textvariable=pw_var, width=20, show="*")
        pw_text.place(relx=0.3, rely=0.53)
        self.config_dict['pw'] = pw_var

        filename_label = tk.Label(RemoteWin, text="File Path:", width=10)
        filename_label.place(relx=0.02, rely=0.75)
        filename_var = tk.StringVar()
        filename_var.set(self.host_filename)
        print(self.host_filename)
        filename_text = tk.Text(RemoteWin, width=30, height=2, wrap='word')
        filename_text.place(relx=0.3, rely=0.70)
        filename_text.insert('end', filename_var.get())
        self.config_dict['filename'] = filename_var

        connect_button = tk.Button(RemoteWin, text="OK", command=self.get_config)
        connect_button.place(relx=0.8, rely=0.4, anchor="center")

        save_host_info_label = tk.Label(RemoteWin, text="Save host", width=10, height=1, anchor='nw')
        save_host_info_label.place(relx=0.81, rely=0.51)
        self.save_host_info_var = tk.IntVar()
        self.save_host_info_var.set(0)
        save_checkbutton = tk.Checkbutton(RemoteWin, variable=self.save_host_info_var)
        save_checkbutton.place(relx=0.73, rely=0.5)

        hostlist_buttom = tk.Button(RemoteWin,  text="Host List",
                                    command=self.HostListInterface)
        hostlist_buttom.place(relx=0.75, rely=0.02)
        RemoteWin.mainloop()

    def HostListInterface(self):

        print('Host List')
        HostListWin = tk.Toplevel()
        HostListWin.title("Remote Host List")
        HostListWin.geometry('400x200')

        self.hostlist_viewer = HostListWin

        self.go_save_host_info()

        # print(self.hostlist_dict)
        self.hostlist_dict = yaml.load(open("./RunConfig/HostList.yaml", 'r', encoding="utf-8"))
        selected_host_var = tk.StringVar()
        hostlist_listbox = tk.Listbox(HostListWin, listvariable=selected_host_var, width=300, height=100)
        self.hostlist = list()
        tmp_list = list(self.hostlist_dict.keys())
        tmp_list.sort()
        for host in tmp_list:
            info_str = "{}   {} : {}   {}".format(self.hostlist_dict[host]["user"],
                                                  self.hostlist_dict[host]["hostip"], self.hostlist_dict[host]["port"],
                                                  self.hostlist_dict[host]["filename"])
            hostlist_listbox.insert('end', info_str)
            self.hostlist.append(host)
        hostlist_listbox.pack()
        self.config_dict["hostlist"] = hostlist_listbox

        selected_host_button = tk.Button(HostListWin, text="OK", command=self.update_interface)
        selected_host_button.place(relx=0.3, rely=0.8)
        delete_host_button = tk.Button(HostListWin, text="Delete", command=self.go_delete_host)
        delete_host_button.place(relx=0.7, rely=0.8)

        HostListWin.mainloop()



