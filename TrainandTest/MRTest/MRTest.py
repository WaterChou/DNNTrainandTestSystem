import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"    # 指定0号GPU

import RunTestModel

pid = os.getpid()
print("\nRemote MR Test, pid ", pid)

print(os.path.abspath(os.path.join(os.getcwd(), ".")))

# test_configs = json.load(open("./RunConfig/TestConfig.jason", "r"))
test_configs = json.load(open("PycharmProjects/Chou_Jnyi/Share/TrainandTest/MRTest/RunConfig/TestConfig.jason"))
# test_configs = json.load(open("code/TrainandTest/MRTest/RunConfig/TestConfig.jason"))
test_configs['pid'] = pid
print(test_configs)
# with open("./RunConfig/TestConfig.jason", "w") as f:
#     f.write(json.dumps(test_configs))
with open("PycharmProjects/Chou_Jnyi/Share/TrainandTest/MRTest/RunConfig/TestConfig.jason", "w") as f:
    f.write(json.dumps(test_configs))
# with open("code/TrainandTest/MRTest/RunConfig/TestConfig.jason", "w") as f:
#     f.write(json.dumps(test_configs))

mr_test_dnn = RunTestModel.RunTestModel(test_configs)
mr_test_dnn.run()

test_configs['accuracy'] = mr_test_dnn.acc
test_configs['kmncov'] = mr_test_dnn.kmncov
test_configs['nbcov'] = mr_test_dnn.nbcov
test_configs['sancov'] = mr_test_dnn.sancov
test_configs['tkncov'] = mr_test_dnn.tkncov
test_configs['tknpat'] = mr_test_dnn.tknpat

with open("{}RunConfig/TestConfig.jason".format(test_configs['file_name']), "w") as f:
    f.write(json.dumps(test_configs))

print(test_configs['accuracy'])

print("\n"+"="*16+"End"+"="*16)

