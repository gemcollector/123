import pandas as pd
import numpy as np
import gym
import os
import socket
import time
from io import StringIO
from common import *
from ModelPkg import *
from BackProNN_actor import Actor as acnn
from BackProNN_critic import Critic as ccnn
import json

import psutil

env = gym.make('CartPole-v0')
env.seed(1)

MAX_EPISODE = 1000
learning_rate = 0.01


# for i_episode in range(MAX_EPISODE):
#     s = env.reset()
#     t = 0
#     while True:
#         s_, r, done, info = env.step(1)
#
#         if done:
#             r = -20
#             break
#     print(r)


class Client:
    def __init__(self):
        self.name_node_sock = socket.socket()
        # self.name_node_sock.connect((name_node_host, name_node_port))

    def start(self):
        env_shape, hidden1, hidden2, K = 4, 32, 32, 2
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
        MAX_EPISODE = 1000
        reward_list = []
        request = "start"
        print("Request: {}".format(request))

        socket_num = []
        cpu_list = []
        receivesize_list = []
        sendsize_list = []
        for host_name in host_list:
            data_node_sock = socket.socket()
            try:
                data_node_sock.connect((host_name, data_node_port))
            except ConnectionRefusedError:
                data_node_sock.connect(('thumm05', data_node_port))
            # 让data_node开始运行
            data_node_sock.send(bytes(request, encoding='utf-8'))
            socket_num.append([host_name, data_node_sock])
        count = 0
        while True:
            cpu_list.append(psutil.cpu_percent())
            for one_socket in socket_num:
                node_sock = one_socket[1]

                print('node_sock sent')
                all_msg = bytes()
                #recv_request = 'receive'
                time.sleep(0.0002)

                rec_msg = str(node_sock.recv(BUF_SIZE), encoding='utf-8')
                try:
                    size_msg = int(rec_msg)
                except Exception:
                    msg = rec_msg
                    if (msg == '') or (msg == 'finish'):
                        socket_num.remove(one_socket)
                        node_sock.close()
                    continue

                print('size_msg', size_msg)
                if size_msg > 0:
                    receivesize_list.append(size_msg)
                    recv_request = 'receive'
                    node_sock.send(bytes(recv_request, encoding='utf-8'))

                    aim_size = size_msg
                    now_size = 0
                    while now_size < aim_size:
                        response_msg = node_sock.recv(BUF_SIZE)
                        now_size += len(response_msg)
                        print(now_size)
                        all_msg += response_msg

                #node_sock.send(bytes(recv_request, encoding='utf-8'))
                    count += 1
                    response_msg = str(all_msg, encoding='utf-8')
                    fp = open('string.txt', 'wb')
                    fp.write(all_msg)
                    fp.close()
                    ####接梯度，这里把梯度写出来
                    actor_grad = response_msg.strip().split('|')[0]
                    critic_grad = response_msg.strip().split('|')[1]

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

                            aim_dict['g%s' % ele[0]] = np.array(ele[1]).reshape(np.shape(np.array(weight[ele[0]]))[0],
                                                                           np.shape(np.array(weight[ele[0]]))[1])

                        return aim_dict
                    actor_aim_dict = decode_string(actor_grad, master_actor.weight)
                    critic_aim_dict = decode_string(critic_grad, master_critic.weight)
                #print('this_aim_aict {}'.format(aim_dict))

                    ####然后这里做梯度下降
                    master_actor.gradient_decend(actor_aim_dict, server_choice=one_socket[0])
                    master_critic.gradient_decend(critic_aim_dict, server_choice=one_socket[0])

                    ####打印梯度
                    #print('actor_aim_dict', actor_aim_dict)
                    ####然后发送下降后的权重
                    def convert_string(weight):
                        string_value = str()
                        for key in sorted(weight.keys()):
                            for row in weight[key]:
                                for column in row:
                                    string_value += str(column) + ' '
                            string_value += '\n'
                        return string_value

                    try_actor_weight = convert_string(master_actor.weight)
                    try_critic_weight = convert_string(master_critic.weight)
                    try_all_weight = try_actor_weight + '|' + try_critic_weight
                    try_weight_size = len(bytes(try_all_weight, encoding='utf-8'))
                    print('gover')
                    # weight_order = 'gover'
                    # node_sock.send(bytes(weight_order, encoding='utf-8'))
                    print('grad_size is:', try_weight_size)
                    sendsize_list.append(try_weight_size)
                    time.sleep(0.0002)
                    node_sock.send(bytes(str(try_weight_size), encoding='utf-8'))
                    time.sleep(0.0002)
                    # request3 = str(node_sock.recv(BUF_SIZE), encoding='utf-8')
                    # print('try_request3:', request3)
                    if str(node_sock.recv(BUF_SIZE), encoding='utf-8') == 'receive1':
                        node_sock.send(bytes(str(try_all_weight), encoding='utf-8'))

                    #### data_node接参数，改权重
                    # print(response_msg)
                    # episode_reward = response_msg.split(',')
                    # del episode_reward[-1]
                    # print(episode_reward)
                    # for i in episode_reward:
                    #     reward_list.append(i)

                    #print('node %s end' % one_socket[0])
            # print(len(socket_num))
            if len(socket_num) == 0:
                break

        pd.Series(cpu_list).to_csv('cpu_result_try.csv', index=0)
        pd.Series(sendsize_list).to_csv('send_result_try.csv', index=0)
        pd.Series(receivesize_list).to_csv('receive_result_try.csv', index=0)

        print('My count {}'.format(len(reward_list)))


import sys

argv = sys.argv
argc = len(argv) - 1

action_shape = env.action_space
print(action_shape)

client = Client()
cmd = argv[1]

if cmd == '-start':
    client.start()
else:
    print("Undefined command: {}".format(cmd))
    print("Usage: python client.py <-ls | -copyFromLocal | -copyToLocal | -rm | -format> other_arguments")
