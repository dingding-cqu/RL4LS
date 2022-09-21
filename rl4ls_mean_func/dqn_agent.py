import datetime
import os.path
import random

import torch
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

from memory import Mem
from env import Environment
from wcsp.core.parser import parse

class DQNAgent:
    def __init__(self, model, target_model, optimizer, device='cpu', capacity=10000000, epsilon=.9, scale=10, gamma=.99):
        self.model = model
        self.target_model = target_model
        self.device = device
        self.target_model.load_state_dict(model.state_dict())
        self.model.to(device)
        self.target_model.to(device)
        # self.epsilon = epsilon
        self.scale = scale
        self.memory = Mem(capacity)
        self.optimizer = optimizer
        self.gamma = gamma
        self.max_itr_cnt = 10000

    def train(self, train_list, validation_list, validation=50, batch_size=32, model_path='../rl4ls_0919_models'):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        cnt = 1
        # self.epsilon = 0
        f = open("log.txt", "w")
        while True:
            # print("train cnt: ", cnt, " ", self.epsilon)
            # self.epsilon += 0.05 * int(cnt/100)
            # self.epsilon = min(self.epsilon, 0.9)
            pth = random.choice(train_list)
            all_vars, all_functions = parse(pth, self.scale)
            env = Environment(all_vars, all_functions, in_valid=False)
            total_reward = self._decision_making(env, f)
            losses = []
            for _ in range(50):
                losses.append(self._learn(batch_size))
            print(f'Iteration {cnt}\t {(total_reward * self.scale/len(all_functions)):.2f}\t {sum(losses) / len(losses):.2f}\t{datetime.datetime.now()}')
            f.write(f'Iteration {cnt}\t {(total_reward * self.scale/len(all_functions)):.2f}\t {sum(losses) / len(losses):.2f}\t{datetime.datetime.now()}')

            cnt += 1
            if cnt % validation == 0:
                cost = []
                for vp in validation_list:
                    all_vars, all_functions = parse(vp, self.scale)
                    env = Environment(all_vars, all_functions, in_valid=True)
                    cost.append(self._decision_making(env, f, False)/len(all_functions))
                tag = int(cnt / validation)
                print(f'Validation {tag}\t {(sum(cost) * self.scale / len(cost)):.2f}\t {sum(losses) / len(losses):.2f}\t{datetime.datetime.now()}')
                f.write(f'Validation {tag}\t {(sum(cost) * self.scale / len(cost)):.2f}\t {sum(losses) / len(losses):.2f}\t{datetime.datetime.now()}')

                torch.save(self.model.state_dict(), f'{model_path}/{tag}.pth')

    def _decision_making(self, env, f, training=True):
        x, edge_index, all_func_ind, action_size = env.build_graph()
        max_cnt = action_size*30
        cnt = -1
        best_cost = -1
        while cnt < max_cnt:
            cnt += 1
            s = Data(x=x, edge_index=edge_index, function_idx=all_func_ind)
            q_values = self.model.inference(x.to(self.device), edge_index.to(self.device), action_size, all_func_ind)
            if training:
                q_values.squeeze_()
                pop = torch.nn.functional.softmax(q_values)
                action = random.choices([i for i in range(action_size)], weights=pop.tolist(), k=1)[0]
            else:
                action = q_values.argmax().item()
            r, x, best_cost = env.act(action)
            if r > 0:
                print(max_cnt, cnt, int(r*100)/100)
                f.write(f'{max_cnt}, {cnt}, {int(r*100)/100}')
            if training:
                s_prime = Data(x=x, edge_index=edge_index, function_idx=all_func_ind, action_space=action_size)
                self.memory.add(s, action, r, s_prime, cnt == max_cnt)
        return best_cost

    def _learn(self, batch_size):
        self.optimizer.zero_grad()
        s, a, r, s_prime, done = self.memory.sample(batch_size)
        batch = Batch.from_data_list(s)
        batch.x = batch.x.to(self.device)
        batch.edge_index = batch.edge_index.to(self.device)
        self.model.eval()
        pred = self.model(batch, a)
        targets = []
        for i in range(len(s_prime)):
            if done[i]:
                targets.append(0)
            else:
                s = s_prime[i]
                q_values = self.target_model.inference(s.x.to(self.device), s.edge_index.to(self.device), s.action_space,
                                                       s.function_idx)
                targets.append(q_values.max().item())
        targets = torch.tensor(r, dtype=torch.float32, device=self.device) + self.gamma * torch.tensor(targets, dtype=torch.float32, device=self.device)
        targets.unsqueeze_(1)

        self.model.train()
        loss = F.mse_loss(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self._soft_update()
        return loss.item()

    def _soft_update(self, tau=.0001):
        for t_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            new_param = tau * param.data + (1.0 - tau) * t_param.data
            t_param.data.copy_(new_param)