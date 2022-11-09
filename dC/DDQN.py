import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DuelingDQN import DuelingModel


# class Net(nn.Module):
#     def __init__(self, input, hidden, output):
#         super(Net, self).__init__()
#         self.il = nn.Linear(input, hidden)
#         self.hl1 = nn.Linear(hidden, hidden * 2)
#         self.hl2 = nn.Linear(hidden * 2, hidden * 2)
#         self.hl3 = nn.Linear(hidden * 2, hidden * 2)
#         self.hl4 = nn.Linear(hidden * 2, hidden * 2)
#         self.hl5 = nn.Linear(hidden * 2, hidden)
#         self.ol = nn.Linear(hidden, output)
#
#     def forward(self, x):
#         x = F.relu(self.il(x))
#         x = F.relu(self.hl1(x))
#         x = F.relu(self.hl2(x))
#         x = F.relu(self.hl3(x))
#         x = F.relu(self.hl4(x))
#         x = F.relu(self.hl5(x))
#         x = self.ol(x)
#         return x


class DeepQNetwork():
    def __init__(self, input, actions_cnt, hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=200, batch_size=32, e_greedy_increment=None,
                 ):
        # self.actions = actions
        # self.actions_cnt = len(actions)
        self.actions_cnt = actions_cnt
        self.input = input
        self.hidden = hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter  # 替换神经网络需要迭代的次数
        self.memory_size = memory_size  # 可以保存多少sample
        self.memory_counter = 0
        self.batch_size = batch_size  # 从sample集合中,每次训练sample样品数
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, input * 2 + 2))  # size = 6 <- 输入俩坐标 + action + reward + 输出俩坐标

        self.loss_func = nn.MSELoss()  # 这个以后可以尝试换下其他的

        self._build_net()

    def _build_net(self):
        # self.q_eval = Net(self.input, self.hidden, self.actions_cnt)
        # self.q_target = Net(self.input, self.hidden, self.actions_cnt)
        self.q_eval = DuelingModel()
        self.q_target = DuelingModel()
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)  # RMSprop可以自己了解下是啥哟

    def store_transition(self, state, action, reward, next_state):
        infos = np.array(state).flatten()
        # linfos = list(infos)
        next_infos = np.array(next_state).flatten()
        # lnext_infos = list(next_infos)

        temp = np.array([action, reward, ])
        packet_infos = np.concatenate((infos, temp, next_infos))

        if self.memory_counter == self.memory_size:
            self.memory_counter = 0
        # self.memory[self.memory_counter, :] = np.array(
        #     [state[0], state[1], action, reward, next_state[0], next_state[1]])
        self.memory[self.memory_counter, :] = packet_infos
        self.memory_counter += 1

    def choose_action(self, observation_state, neighbors, mbatch_memory):
        # input = torch.Tensor([[observation_state[0], observation_state[1]]])
        input = torch.Tensor([observation_state])  # test input
        ninput = np.array(input)
        ninput.resize((9, 9))
        nninput = torch.Tensor([ninput])
        # print(input)
        action = -1
        if np.random.uniform() < self.epsilon:
            actions_value = self.q_eval(nninput)
            values = actions_value.data.numpy()
            # print(values)
            temp = []
            lv = list(values.__getitem__(0))
            cnt = 1
            # print(values)
            for v in lv:
                temp.append({'cnt': cnt, 'reward': v})
                cnt += 1
            dtemp = sorted(temp, key=lambda x: x['reward'])
            print(dtemp)
            for v in dtemp:
                if mbatch_memory['Node' + str(v['cnt'])]['id'] in neighbors:
                    action = v['cnt']
                    break
                else:
                    continue
            # action = np.argmin(values)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            # print(self.memory_counter, self.batch_size)
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        old = batch_memory[:, -self.input:]
        iold = []
        for line in old:
            nline = np.array(line)
            nline.resize((9, 9))
            iold.append(list(nline))
        # old.resize((6, 6))
        inew = []
        new = batch_memory[:, :self.input]
        for line in new:
            nline = np.array(line)
            nline.resize((9, 9))
            inew.append(list(nline))
        # new.resize((6, 6))

        # q_next = self.q_target(torch.Tensor(np.array([iold])))  # 老的
        # q_eval = self.q_eval(torch.Tensor(np.array([inew])))
        #
        # q_target = torch.Tensor(q_eval.data.numpy().copy())
        #
        # batch_index = np.arange(self.batch_size, dtype=np.int32)  # 仅仅是 0-31 index下标
        # eval_act_index = batch_memory[:, self.input].astype(int)  # 较老的但也不是很老的做了什么动作
        # reward = torch.Tensor(batch_memory[:, self.input + 1])  # 取出来较老的但也不是很老的奖励是多少
        # print('reward', reward)
        #
        # print("q_target[batch_index, eval_act_index]", q_target[batch_index, eval_act_index])
        # print("torch.max(q_next, 1)[0]", torch.max(q_next, 1)[0])
        # q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]  # 算老的reward
        # # q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]  # 算老的reward
        #
        # loss = self.loss_func(q_eval, q_target)  # 和新的比较差了多少,你会发现其他没采取动作的地方相减为0
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        count = 0
        for o, n in zip(iold, inew):
            q_next = self.q_target(torch.Tensor(np.array([o])))  # 老的
            q_eval = self.q_eval(torch.Tensor(np.array([n])))

            q_target = torch.Tensor(q_eval.data.numpy().copy())

            # batch_index = np.arange(self.batch_size, dtype=np.int32)  # 仅仅是 0-31 index下标
            # eval_act_index = batch_memory[:, self.input].astype(int)  # 较老的但也不是很老的做了什么动作
            # reward = torch.Tensor(batch_memory[:, self.input + 1])  # 取出来较老的但也不是很老的奖励是多少

            # print('batch_memory', batch_memory)
            batch_index = count  # 仅仅是 0-31 index下标
            count += 1
            # print("batch_index", batch_index)
            # print('batch_memory ->', batch_memory, 'batch_index ->', batch_index, 'self.input ->', self.input)
            # print('batch_memory[batch_index, self.input] ->', batch_memory[batch_index, self.input])
            eval_act_index = batch_memory[batch_index, self.input].astype(int)  # 较老的但也不是很老的做了什么动作
            # print(len(batch_memory[batch_index,]), '<-len  self.input ->', self.input)
            # print("eval_act_index", batch_memory[batch_index, self.input], eval_act_index)
            reward = torch.Tensor([batch_memory[batch_index, self.input + 1]])  # 取出来较老的但也不是很老的奖励是多少
            # print("reward", reward)

            # print("q_target[batch_index, eval_act_index]", q_target[0, eval_act_index])
            # print("torch.max(q_next, 1)[0]", torch.max(q_next, 1)[0])
            # print('eval_act_index ->', eval_act_index)
            q_target[0, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]  # 算老的reward
            # q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]  # 算老的reward

            loss = self.loss_func(q_eval, q_target)  # 和新的比较差了多少,你会发现其他没采取动作的地方相减为0
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        # print(self.epsilon)
