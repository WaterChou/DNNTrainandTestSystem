import paramiko
import os


class SSHConnection(object):
    def __init__(self, host, port, username, password):
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._transport = None
        self._sftp = None
        self._client = None
        self._connect()  # 建立连接

    def _connect(self):
        transport = paramiko.Transport((self._host, self._port))
        transport.connect(username=self._username, password=self._password)
        self._transport = transport

    # 下载
    def download(self, remotepath, localpath):
        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        self._sftp.get(remotepath, localpath)

    # 上传
    def put(self, localpath, remotepath):
        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        self._sftp.put(localpath, remotepath)

    # 执行命令
    def exec_command(self, command):
        if self._client is None:
            self._client = paramiko.SSHClient()
            self._client._transport = self._transport
        stdin, stdout, stderr = self._client.exec_command(command, get_pty=True)
        ###方法一： 实时返回输出###
        while not stdout.channel.exit_status_ready():
            result = stdout.readline()
            print(result)
            # yield result
            # 由于在退出时，stdout还是会有一次输出，因此需要单独处理，处理完之后，就可以跳出了
            if stdout.channel.exit_status_ready():
                a = stdout.readlines()
                print(a)
                break
        ###############################

        ###方法二： 实时返回输出###
        # def line_buffered(f):
        #     line_buf = ""
        #     while not f.channel.exit_status_ready():
        #         # print(type(line_buf))
        #         # print(type(f.read(1)))
        #         line_buf += f.read(1).decode()
        #         if line_buf.endswith('\n'):
        #             yield line_buf
        #             line_buf = ''
        #
        # for l in line_buffered(stdout):
        #     print(l)
        ###############################

    def close(self):
        if self._transport:
            self._transport.close()
        if self._client:
            self._client.close()


def exec_command(client, command):

    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    ###方法一： 实时返回输出###
    while not stdout.channel.exit_status_ready():
        result = stdout.readline()
        print(result)
        # yield result
        # 由于在退出时，stdout还是会有一次输出，因此需要单独处理，处理完之后，就可以跳出了
        if stdout.channel.exit_status_ready():
            a = stdout.readlines()
            print(a)
            break


if __name__ == "__main__":

    # client = paramiko.SSHClient()
    # client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # client.connect('192.168.3.52', 22, username='xiaohui', password='123456')
    # cm = "echo 'export PATH=$PATH:/home/xiaohui/anaconda3/bin' >> ~/.profile"
    # cm = ". ~/.profile; python"
    # cm = "anaconda3/envs/tf/bin/python; python MRTrain"
    # stdin, stdout, stderr = client.exec_command(cm)
    # print(stdout.read()) "ldconfig /usr/local/cuda/lib64;"\
    conn = SSHConnection('192.168.3.80', 22, username='choujnyi', password='123456')
    # cm = "anaconda3/envs/tf/bin/python " \
    #      "code/TrainandTest/MRTest/MRTest.py"

    # conn.exec_command(cm)
    rm_path = "code/TrainandTest/MRTest/RunConfig/TestConfig.jason"
    lc_path = "./RunConfig/TestConfig.jason"
    # print(os.path.isfile(lc_path))
    conn.put(lc_path, rm_path)
    # conn.download(rm_path, lc_path)
    conn.close()
