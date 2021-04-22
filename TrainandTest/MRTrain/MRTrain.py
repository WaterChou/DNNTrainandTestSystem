import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import time

import RunTrainModel

print(os.path.abspath(os.path.join(os.getcwd(),".")))

pid = os.getpid()
print("Remote MR Train, pid ", pid)
# train_configs = json.load(open("PycharmProjects/Chou_Jnyi/Share/TrainandTest/MRTrain/RunConfig/TrainConfig.jason"))
train_configs = json.load(open("./MRTrain/RunConfig/TrainConfig.jason", "r"))
train_configs['pid'] = pid
# print(train_configs)
with open("./MRTrain/RunConfig/TrainConfig.jason", "w") as f:
    f.write(json.dumps(train_configs))
# with open("PycharmProjects/Chou_Jnyi/Share/TrainandTest/MRTrain/RunConfig/TrainConfig.jason", "w") as f:
#     f.write(json.dumps(train_configs))

if train_configs['dataset'] == "MNIST":
    train_configs['drop_label'] = []
elif train_configs['dataset'] == "CIFAR10":
    train_configs['drop_label'] = []
elif train_configs['drop_label'] == "MNIST(0,1,8)":
    train_configs['drop_label'] = [2, 3, 4, 5, 6, 7, 9]
elif train_configs['drop_label'] == "MNIST(0,1,3,8)":
    train_configs['drop_label'] = [2, 4, 5, 6, 7, 9]


mr_train_dnn = RunTrainModel.RunDataset(train_configs)
mr_train_dnn.run()
train_configs['accuracy'] = mr_train_dnn.acc

# print('\nTraining...')
# for r in range(train_configs['runtime']):
#     print("Run times =  {}/{}".format(r+1, train_configs['runtime']))
#     for e in range(train_configs['epochs']):
#         print("Epoch = {}/{}".format(e+1, train_configs['epochs']))
#         time.sleep(1)
# print("\nEvaluating on test set")
# print("\nacc={}".format(0.9880001))
# train_configs['accuracy'] = 0.9880001

with open("{}RunConfig/TrainConfig.jason".format(train_configs['file_name']), "w") as f:
    f.write(json.dumps(train_configs))

print("\n"+"="*16+"End"+"="*16)

