import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from delta import delta_operator
from substract_min import substract_min_operator
from zigzag import zigzag_encode
from RLE import encode_rle
from DictionaryInt import dictionaryEncoding
import gzip
import snappy
import lz4.frame
import lzma
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

def generate_random_time_series(length, low=minbound, high=maxbound):
    time_series = np.random.randint(low, high, size=length)
    return time_series

def original(seq):
    return seq

# 定义环境
class EncodingEnvironment:
    def __init__(self, initial_sequence):
        self.sequence = initial_sequence
        self.current_encoding = None
        self.action_space = [original, delta_operator, substract_min_operator, zigzag_encode, encode_rle, dictionaryEncoding] 

    def step(self, action):
        # 应用编码算子
        if action == encode_rle or action == dictionaryEncoding:
            self.current_encoding = action(self.sequence)[0] + [0] * (len(self.sequence) - len(action(self.sequence)[0]))
        else:
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

    compressed_data = action(encoding)

    # 使用 gzip 进行压缩
    data_gzip = gzip.compress(pickle.dumps(encoding))
    datasize_gzip = len(data_gzip)
    compressed_data_gzip = gzip.compress(pickle.dumps(compressed_data))
    compressed_datasize_gzip = len(compressed_data_gzip)

    # 使用 snappy 进行压缩
    data_snappy = snappy.compress(pickle.dumps(encoding))
    datasize_snappy = len(data_snappy)
    compressed_data_snappy = snappy.compress(pickle.dumps(compressed_data))
    compressed_datasize_snappy = len(compressed_data_snappy)

    # 使用 lz4 进行压缩
    data_lz4 = lz4.frame.compress(pickle.dumps(encoding))
    datasize_lz4 = len(data_lz4)
    compressed_data_lz4 = lz4.frame.compress(pickle.dumps(compressed_data))
    compressed_datasize_lz4 = len(compressed_data_lz4)

    # 使用 bp 进行压缩
    datasize_bp = bp(encoding)
    compressed_datasize_bp = bp(compressed_data)

    # 使用 xz 压缩
    data_xz = lzma.compress(pickle.dumps(encoding), format=lzma.FORMAT_XZ)
    datasize_xz = len(data_xz)
    compressed_data_xz = lzma.compress(pickle.dumps(compressed_data), format=lzma.FORMAT_XZ)
    compressed_datasize_xz = len(compressed_data_xz)

    # 计算奖励，可以根据具体情况进行调整
    reward_gzip = datasize_gzip - compressed_datasize_gzip
    reward_snappy = datasize_snappy - compressed_datasize_snappy
    reward_lz4 = datasize_lz4 - compressed_datasize_lz4
    reward_bp = datasize_bp - compressed_datasize_bp
    reward_xz = datasize_xz - compressed_datasize_xz

    # 返回使用不同压缩算法得到的奖励，你可以根据具体情况选择使用哪个算法
    return max(reward_gzip, reward_snappy, reward_lz4, reward_bp, reward_xz)


def bp(encoding):
    compressed_size_bp = 999999
    if min(encoding) >= 0:
        compressed_size_bp = 0
        segmented_sequence = [encoding[i:i+8] for i in range(0, len(encoding), 8)]

        for segment in segmented_sequence:
            width = math.ceil(math.log(max(segment)))
            compressed_size_bp = compressed_size_bp + width*len(segment) / 8
    return compressed_size_bp

# 主循环
def main():
    dataset = [generate_random_time_series(sequence_length) for _ in range(dataset_size)]

    # 初始化环境和代理
    initial_sequence = dataset[0]  # 使用数据集的第一个序列作为初始序列
    env = EncodingEnvironment(initial_sequence)
    agent = RLAgent(state_size=len(initial_sequence), action_size=len(env.action_space))

    # 记录每个编码方案的推荐次数
    action_counts = {action: 0 for action in env.action_space}

    # 训练代理
    count = 0
    for sequence in dataset:
        count = count + 1
        print(count)
        state = np.reshape(sequence, [1, len(sequence)])
        action_index = agent.act(state)
        action = env.action_space[action_index]

        # 记录每个编码方案的推荐次数
        action_counts[action] += 1
        print(action)

        next_encoding, reward = env.step(action)
        next_state = np.reshape(next_encoding, [1, len(next_encoding)])
        agent.train(state, action_index, reward, next_state)

    # 保存模型
    agent.save_model('model.h5')
    
    # 打印每个编码方案的推荐次数
    print("Action Counts:")
    for action, count in action_counts.items():
        print(f"{action}: {count}")

if __name__ == "__main__":
    main()
