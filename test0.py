import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from delta_operator import delta_operator
from substract_min_operator import substract_min_operator
from zigzag import zigzag_encode
from RLE import encode_rle
import gzip
import snappy
import lz4.frame
import pickle
import math
import os
import csv
import time

# print(os.getcwd())

dataset_size = 10000
sequence_length = 1000
minbound = -10000
maxbound = 10000

def original(seq):
    return seq

# 定义环境
class EncodingEnvironment:
    def __init__(self, initial_sequence):
        self.sequence = initial_sequence
        self.current_encoding = None
        self.action_space = [delta_operator, substract_min_operator, zigzag_encode, original] 

    def step(self, action):
        # 应用编码算子
        self.current_encoding = action(self.sequence)
        # 计算奖励（需要根据任务具体情况进行定义）
        reward = calculate_reward(self.current_encoding, action)
        return self.current_encoding, reward

# 定义强化学习代理
class RLAgent:
    def __init__(self, state_size, action_size, c = 10000):
        self.state_size = state_size
        self.action_size = action_size
        # self.epsilon = 0.1
        self.gamma = 0.95  # 折扣因子
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.action_counts = np.zeros(action_size)  # 记录每个动作的选择次数
        self.total_counts = 0  # 记录总的动作选择次数
        self.c = c  # UCB 算法的探索参数

    def build_model(self):
        model = Sequential()
        model.add(Dense(240, input_dim=self.state_size, activation='relu'))
        model.add(Dense(240, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        # 使用 epsilon-greedy
        # if np.random.rand() < self.epsilon:
        #     # 以 epsilon 概率选择随机动作
        #     return np.random.choice(self.action_size)
        # else:
        #     # 否则，根据当前策略选择动作
        #     q_values = self.model.predict(state)
        #     return np.argmax(q_values[0])
        # 使用 UCB 算法进行动作选择
        q_values = self.model.predict(state)[0]
        ucb_values = q_values + self.c * np.sqrt(np.log(self.total_counts + 1) / (self.action_counts + 1e-8))
        return np.argmax(ucb_values)

    def train(self, state, action, reward, next_state):
        target = reward + self.gamma * np.amax(self.model.predict(next_state))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        self.action_counts[action] += 1
        self.total_counts += 1

    def save_model(self, file_path):
        # 保存模型
        self.model.save(file_path)

    def load_model(self, file_path):
        # 加载模型
        self.model = tf.keras.models.load_model(file_path)

# 定义奖励函数
def calculate_reward(encoding,action):
    if action != original:
        return 0
    original_size = sequence_length * 32  # 假设原始序列每个元素占 32 位

    # 使用 gzip 进行压缩
    compressed_data_gzip = gzip.compress(pickle.dumps(encoding))
    compressed_size_gzip = len(compressed_data_gzip)

    # 使用 snappy 进行压缩
    compressed_data_snappy = snappy.compress(pickle.dumps(encoding))
    compressed_size_snappy = len(compressed_data_snappy)

    # 使用 lz4 进行压缩
    compressed_data_lz4 = lz4.frame.compress(pickle.dumps(encoding))
    compressed_size_lz4 = len(compressed_data_lz4)

    # 使用 bp 进行压缩
    compressed_size_bp = 999999
    if min(encoding) >= 0:
        compressed_size_bp = 0
        segmented_sequence = [encoding[i:i+8] for i in range(0, len(encoding), 8)]

        for segment in segmented_sequence:
            width = math.ceil(math.log(max(segment)))
            compressed_size_bp = compressed_size_bp + width*len(segment) / 8

    # 计算奖励，可以根据具体情况进行调整
    reward_gzip = original_size / compressed_size_gzip
    reward_snappy = original_size / compressed_size_snappy
    reward_lz4 = original_size / compressed_size_lz4
    reward_bp = original_size / compressed_size_bp

    # 返回使用不同压缩算法得到的奖励，你可以根据具体情况选择使用哪个算法
    return -min(compressed_size_gzip, compressed_size_lz4, compressed_size_snappy, compressed_size_bp)

def test():
    ppath = r'trans_data/'
    dataset_names = ['CS-Sensors','Cyber-Vehicle','EPM-Education','GW-Magnetic','Metro-Traffic','Nifty-Stocks','TH-Climate','TY-Fuel','TY-Transport','USGS-Earthquakes','Vehicle-Charge','YZ-Electricity']
    count1 = 0
    count2 = 0
    data = [['datasetName','compressed_size','model_size','opt_size','model_time','opt_time']]
    results = []
    for dataset_name in dataset_names:
        dir_path = ppath + dataset_name
        file_names = os.listdir(dir_path)
        compressed_size_bp = 0
        pre_size = 0
        min_size = 0
        pre_time = 0
        min_time = 0
        for file_name in file_names:
            file1_path = dir_path + '/' + file_name
            # file1_path = 'GW-Magnetic/syndata_magnetic0.csv'
            # file1_path = 'USGS-Earthquakes/syndata_earthquakes0.csv'
            df = pd.read_csv(file1_path)
            columns = df.columns[1:2]
            data1 = df[columns]
            array = data1.to_numpy()
            chunked_array = [array[i:i+1000] for i in range(0, len(array), 1000)]
            for array1 in chunked_array:
                if len(array1) < 1000:
                    continue
                array1 = zigzag_encode(array1)

                # 使用 gzip 进行压缩
                compressed_data_gzip = gzip.compress(pickle.dumps(array1))
                compressed_size_gzip = len(compressed_data_gzip)

                # 使用 snappy 进行压缩
                compressed_data_snappy = snappy.compress(pickle.dumps(array1))
                compressed_size_snappy = len(compressed_data_snappy)

                # 使用 lz4 进行压缩
                compressed_data_lz4 = lz4.frame.compress(pickle.dumps(array1))
                compressed_size_lz4 = len(compressed_data_lz4)

                compressed_size_bp = compressed_size_bp + min(compressed_size_gzip, compressed_size_lz4, compressed_size_snappy)

                # segmented_sequence = [array1_ts2diff[i:i+8] for i in range(0, len(array1_ts2diff), 8)]

                # for segment in segmented_sequence:
                #     if max(segment) > 1:
                #         width = math.ceil(math.log(max(segment)))
                #     else:
                #         width = 1
                #     compressed_size_bp = compressed_size_bp + width*len(segment)
                # print(compressed_size_bp)

                # 加载模型
                env = EncodingEnvironment(array1)
                agent = RLAgent(state_size=1000, action_size=4)
                agent.load_model('model copy.h5')
                
                # test_sequence = generate_random_time_series(sequence_length, low=minbound, high=maxbound)
                test_state = np.reshape(array1, [1, len(array1)])
                test_action_index = agent.act(test_state)
                test_action = env.action_space[test_action_index]
                # print(test_action)
                s = time.time()
                _, test_reward = env.step(test_action)

                # 使用 gzip 进行压缩
                compressed_data_gzip = gzip.compress(pickle.dumps(_))
                compressed_size_gzip = len(compressed_data_gzip)

                # 使用 snappy 进行压缩
                compressed_data_snappy = snappy.compress(pickle.dumps(_))
                compressed_size_snappy = len(compressed_data_snappy)

                # 使用 lz4 进行压缩
                compressed_data_lz4 = lz4.frame.compress(pickle.dumps(_))
                compressed_size_lz4 = len(compressed_data_lz4)

                pre_size = pre_size + min(compressed_size_gzip, compressed_size_lz4, compressed_size_snappy)
                pre_time = pre_time + time.time() - s


                # print(pre_size)

                # 遍历查找最佳编码
                action_space = [delta_operator, substract_min_operator, zigzag_encode, original] 
                s = time.time()
                for action in action_space:
                    test_state = np.reshape(array1, [1, len(array1)])
                    test_action_index = agent.act(test_state)
                    test_action = env.action_space[test_action_index]
                    # print(test_action)
                    _, test_reward = env.step(test_action)

                    # 使用 gzip 进行压缩
                    compressed_data_gzip = gzip.compress(pickle.dumps(_))
                    compressed_size_gzip = len(compressed_data_gzip)

                    # 使用 snappy 进行压缩
                    compressed_data_snappy = snappy.compress(pickle.dumps(_))
                    compressed_size_snappy = len(compressed_data_snappy)

                    # 使用 lz4 进行压缩
                    compressed_data_lz4 = lz4.frame.compress(pickle.dumps(_))
                    compressed_size_lz4 = len(compressed_data_lz4)

                    min_size1 = min(compressed_size_gzip, compressed_size_lz4, compressed_size_snappy)
                min_size = min_size + min_size1
                min_time = min_time + time.time() - s

    #         print(min_size)
    #         if min_size == pre_size:
    #             count1 = count1 + 1
    #         else:
    #             count2 = count2 + 1
    # print(count1)
    # print(count2)
        data.append([dataset_name,compressed_size_bp,pre_size,min_size,pre_time,min_time])
        results.append([pre_size,min_size,pre_time,min_time])
        print(compressed_size_bp)
        print(pre_size)
        print(min_size)

    with open('result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
    bar_width = 0.15

    x = np.arange(len(dataset_names))
    results = np.array(results)
    plt.bar(x, results[:, 0], width=bar_width, label="model")
    plt.bar(x + bar_width, results[:, 1], width=bar_width, label="opt")
    plt.xlabel('Datasets')
    plt.ylabel('Results')
    plt.title('Results by Dataset and Algorithm')
    plt.xticks(x + bar_width * (2 - 1) / 2, dataset_names, rotation = 45, ha='right')
    plt.legend()
    plt.show()
    bar_width = 0.15
    x = np.arange(len(dataset_names))
    results = np.array(results)
    plt.bar(x, results[:, 2], width=bar_width, label="model")
    plt.bar(x + bar_width, results[:, 3], width=bar_width, label="opt")
    plt.xlabel('Datasets')
    plt.ylabel('Time')
    plt.title('Time by Dataset and Algorithm')
    plt.xticks(x + bar_width * (2 - 1) / 2, dataset_names, rotation = 45, ha='right')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()
