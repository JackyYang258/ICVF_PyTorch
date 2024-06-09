import pickle

# 定义要读取的pkl文件的路径
file_path = 'experiment_output/icvf/icvf/icvf_antmaze-large-diverse-v2_20240609_152731/params.pkl'

# 打开并读取pkl文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印读取的数据
print(data)
