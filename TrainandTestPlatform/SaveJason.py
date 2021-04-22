"""
将字典对象保存为Json文件
"""

import os
import json


class SaveJson:

    def save_file(self, path, file_name, item):

        # 先将字典对象转化为可写入文本的字符串
        item = json.dumps(item)

        try:
            os.makedirs(path, exist_ok=True)
            with open(path+file_name, "w", encoding='utf-8') as f:
                f.write(item)
                print("^_^ write success")
        except Exception as e:
            print("write error==>", e)

    def read_file(self, file_name):
        return  json.load(open(file_name))