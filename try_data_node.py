import pandas as pd
import numpy as np
import gym
import os
import socket
import time
from io import StringIO
from common import *

import os
import socket
import pandas as pd
import json
from common import *
import pickle
import os.path

from ModelPkg import *
from BackProNN_actor import Actor as acnn
from BackProNN_critic import Critic as ccnn
import matplotlib.pyplot as plt

import psutil
import hashlib

MAX_EPISODE = 1000
UPDATE_ITER = 5
GAMMA = 0.9

env = gym.make('CartPole-v0')
env.seed(1)


######在这里把神经网络创建好
class C_Network:
    def __init__(self, s, a):
        self.s = s
        self.a = a
        self.v_target = v_target

    def _build_net(self):
        pass


class A_Network:
    def __init__(self, s):
        self.s = s
        #####损失函数是td_error * log_prob
        self.exp_v = exp_v


# DataNode支持的指令有:
# 1. load 加载数据块
# 2. store 保存数据块
# 3. rm 删除数据块
# 4. format 删除所有数据块

class DataNode:
    def __init__(self):
        pass


    def run(self):
        # 创建一个监听的socket
        listen_fd = socket.socket()
        try:
            # 监听端口
            listen_fd.bind(("0.0.0.0", data_node_port))
            listen_fd.listen(5)
            print("Data node started")
            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_fd.accept()
                print("Received request from {}".format(addr))

                try:
                    # 获取请求方发送的指令
                    # request = str(sock_fd.recv(BUF_SIZE), encoding='utf-8')
                    # request = request.split()  # 指令之间使用空白符分割
                    # print(request)
                    request = pickle.loads(sock_fd.recv(BUF_SIZE))
                    print('request:', request)

                    cmd = request[0]  # 指令第一个为指令类型

                    if cmd == "start":  # 开始游戏
                        response = self.start(sock_fd)
                        print(response)
                    else:
                        response = "Undefined command: " + " ".join(request)

                    sock_fd.send(bytes(response, encoding='utf-8'))
                except KeyboardInterrupt:
                    break
                finally:
                    sock_fd.close()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            listen_fd.close()

    def start(self, sock_fd):
        response_all = str()
        env_shape, hidden1, hidden2, K = 4, 32, 32, 2
        global_reward = []
        master_actor = acnn(weight_dict={
            'W1': np.random.randn(env_shape, hidden1),
            'b1': np.random.randn(hidden1, 1),

            'W2': np.random.randn(hidden1, hidden2),
            'b2': np.random.randn(hidden2, 1),

            'W3': np.random.randn(hidden2, K),
            'b3': np.random.randn(K, 1)
        })
        master_critic = ccnn(weight_dict={
            'W1': np.random.randn(env_shape, hidden1),
            'b1': np.random.randn(hidden1, 1),

            'W2': np.random.randn(hidden1, hidden2),
            'b2': np.random.randn(hidden2, 1),

            'W3': np.random.randn(hidden2, 1),
            'b3': np.random.randn(1, 1)
        })
        cpu_list = []
        receivesize_list = []
        sendsize_list = []
        for i_episode in range(MAX_EPISODE):
            print("No. {}".format(i_episode))
            # print('cpu_percent:', psutil.cpu_percent())
            cpu_list.append(psutil.cpu_percent())

            s = env.reset()
            t = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            total_step = 0
            all_reward = 0
            while True:
                total_step += 1
                a = master_actor.choose_action(s)
                s_, r, done, info = env.step(a)
                all_reward += r
                if done:
                    r = -5
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = master_critic.get_v(s_)  # (这个值应该由神经网络来输出
                        #print('v_s_:', v_s_)
                        # v_s_ = self.master_critic.forward_result['F3']
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                        buffer_v_target)
                    # 把buffer扔到神经网络的参数到中去获取全局的参数
                    data_critic = dict()
                    data_critic["X"] = buffer_s
                    data_critic["Y"] = buffer_v_target

                    master_critic.forward(data_critic)

                    # def dict_to_list(a):
                    #     return np.array(list(a.values()))
                    #
                    # if pd.isna(dict_to_list(master_critic.forward_result)).any():
                    #     print('master_critic.forward_result has NaN')
                    #     exit(0)
                    #
                    td_error = data_critic['Y'] - master_critic.forward_result['F5']
                    #print('td_error:', td_error)
                    master_critic.backward(data_critic)
                    data_actor = dict()
                    data_actor["X"] = buffer_s
                    data_actor["Y"] = buffer_a
                    data_actor["reward"] = td_error
                    master_actor.forward(data_actor)  # 执行一遍foward
                    #print('aaaaaaaaaaaaaa')
                    master_actor.backward(data_actor)  # 梯度都在里面了

                    #df = pd.DataFrame.from_dict(self.master_actor.backward_result)
                    #df.to_csv('df.csv', index = 0)

                    def convert_string(weight):
                        string_value = str()
                        for key in sorted(weight.keys()):
                            for row in weight[key]:
                                for column in row:
                                    string_value += str(column) + ' '
                            string_value += '\n'
                        return string_value

                    actor_gradient = convert_string(master_actor.backward_result)
                    critic_gradient = convert_string(master_critic.backward_result)
                    all_gradient = actor_gradient + '|' + critic_gradient
                    # 所有的梯度x
                    #request2 = str(sock_fd.recv(BUF_SIZE), encoding='utf-8')
                    #print('data_node received request:', request2)
                    grad_size = len(bytes(all_gradient, encoding='utf-8'))
                    #print('grad_size is:', grad_size)
                    sendsize_list.append(grad_size)
                    sock_fd.send(bytes(str(grad_size), encoding='utf-8'))
                    time.sleep(0.0002)
                    request2 = str(sock_fd.recv(BUF_SIZE), encoding='utf-8')
                    if request2 == 'receive':
                        # print('received')
                        sock_fd.send(bytes(all_gradient, encoding='utf-8'))

                    buffer_s, buffer_a, buffer_r = [], [], []
                    # time.sleep(0.0002)
                    # 向master要主网络的参数
                    try:
                        weight_msg = int(str(sock_fd.recv(BUF_SIZE), encoding='utf-8'))
                    except Exception as e:
                        continue
                    #print('I get weight_msg', weight_msg)
                    if weight_msg > 0:
                        recv_request = 'receive1'
                        sock_fd.send(bytes(recv_request, encoding='utf-8'))

                        all_msg = bytes()
                        aim_size = weight_msg
                        now_size = 0
                        while now_size < aim_size:
                            response_msg = sock_fd.recv(BUF_SIZE)
                            now_size += len(response_msg)
                            #print(now_size)
                            all_msg += response_msg
                        receivesize_list.append(now_size)
                        response_msg = str(all_msg, encoding='utf-8')
                        ####接权重
                        actor_weight = response_msg.strip().split('|')[0]
                        critic_weight = response_msg.strip().split('|')[1]

                        ### decode
                        def decode_string(string_value, weight):
                            string_value_list = string_value.strip().split('\n')
                            aim_weight_list = []
                            for numbers in string_value_list:
                                numbers_list = np.array([float(x) for x in numbers.strip().split(' ')])
                                aim_weight_list.append(numbers_list)

                            # print(aim_weight_list)
                            aim_dict = {}
                            #print(weight.items())
                            for ele in zip(sorted(weight.keys()), aim_weight_list):
                                #print(ele[0])
                                # print('ele[0][1].shape', ele[0][1.shape)
                                # print('np.array(ele[1]).shape', np.array(ele[1]).shape)

                                aim_dict['%s' % ele[0]] = np.array(ele[1]).reshape(np.shape(np.array(weight[ele[0]]))[0],
                                                                               np.shape(np.array(weight[ele[0]]))[1])

                            return aim_dict

                        actor_aim_dict = decode_string(actor_weight, master_actor.weight)
                        critic_aim_dict = decode_string(critic_weight, master_critic.weight)

                        master_actor.weight = actor_aim_dict
                        master_critic.weight = critic_aim_dict

                s = s_
                total_step += 1

                if done:
                    #print('all_reward', all_reward)
                    if len(global_reward) == 0:
                        global_reward.append(all_reward)
                    else:
                        global_reward.append(0.99 * global_reward[-1] + 0.01 * all_reward)
                        print(global_reward[-1])
                    break

            # response_all = response_all + (str(r)+',')
            # request2 = str(sock_fd.recv(BUF_SIZE), encoding='utf-8')
            # if request2 == 'receive':
            #     sock_fd.send(bytes(response_all, encoding='utf-8'))
            #     response_all= str()
        x = np.arange(len(global_reward))
        pd.Series(global_reward).to_csv('RL_result.csv', index = 0)
        plt.plot(x, global_reward)
        plt.savefig('worker.png')

        pd.Series(cpu_list).to_csv('cpu_result.csv', index=0)
        pd.Series(sendsize_list).to_csv('send_result.csv', index=0)
        pd.Series(receivesize_list).to_csv('receive_result.csv', index=0)

        response = "finish"
        # response = "yes!"
        return response


# 创建DataNode对象并启动
data_node = DataNode()
data_node.run()


