import copy
import random
from collections import  namedtuple
import torch
from wcsp.core.utility import transpose

x_embed = [1, 0, 0] # current_assignment, current_best_assignment
c_embed = [0, 1, 0, 0] # cost
f_embed = [0, 0, 1, 0, 0]

class Environment:
    def __init__(self, all_vars, all_functions, in_valid=True, device='cpu'):
        self.ind2var_val = dict()
        self.dom_size = dict()
        for name, dom in all_vars:
            self.dom_size[name] = dom
        self.device = device
        self.adj_list = dict()
        self.function_table = dict()
        for data, v1, v2 in all_functions:
            if v1 not in self.adj_list:
                self.adj_list[v1] = list()
                self.function_table[v1] = dict()
            if v2 not in self.adj_list:
                self.adj_list[v2] = list()
                self.function_table[v2] = dict()
            self.function_table[v1][v2] = data
            self.function_table[v2][v1] = transpose(data)
            self.adj_list[v1].append(v2)
            self.adj_list[v2].append(v1)
        self.all_functions = all_functions
        if in_valid:
            self.assignment = {var:0 for var in self.dom_size.keys()}
        else:
            self.assignment = {var:int(random.random()*self.dom_size[var]) for var in self.dom_size.keys()}
        self.best_assignment = dict(self.assignment)
        self.best_cost = self.total_cost(self.best_assignment)
        self.current_cost = self.best_cost

    def total_cost(self, assignment):
        cost = 0
        for data, v1, v2 in self.all_functions:
            cost += data[assignment[v1]][assignment[v2]]
        return cost

    def act(self, assign_ind):
        var, val = self.ind2var_val[assign_ind]
        # old_cost = self.current_cost
        old_val = self.assignment[var]
        self.assignment[var] = val
        self.current_cost = self.total_cost(self.assignment)
        if self.current_cost >= self.best_cost:
            reward = 0
        else:
            reward = self.best_cost - self.current_cost
        if self.current_cost < self.best_cost:
            self.best_cost = self.current_cost
            self.best_assignment = dict(self.assignment)

        if old_val != val:
            old_app_feat, app_feat = [0, 0], [1, 0]
            if self.best_assignment[var] == old_val:
                old_app_feat[1] = 1
            elif self.best_assignment[var] == val:
                app_feat[1] = 1
            self.x[assign_ind+old_val-val][-2:] = old_app_feat
            self.x[assign_ind][-2:] = app_feat
        return reward, torch.tensor(self.x, dtype=torch.float32, device=self.device), self.best_cost

    def build_graph(self):
        x = []
        edge_index = [[], []]
        src, dest = edge_index

        var2ind = dict()
        #      1. assignment nodes
        for var in self.dom_size.keys():
            var2ind[var] = len(x)
            for val in range(self.dom_size[var]):
                self.ind2var_val[len(x)] = (var, val)
                app_feat = [0, 0]
                if self.assignment[var] == val:
                    app_feat[0] = 1
                if self.best_assignment[var] == val:
                    app_feat[1] = 1
                assign_feat = x_embed + app_feat
                x.append(assign_feat)
        assign_len = len(x)

        all_func_ind = []
        #      2. function nodes and function cost nodes
        for data, v1, v2 in self.all_functions:
            func_ind = len(x)
            all_func_ind.append(func_ind)
            x.append(f_embed)
            v1_ind, v2_ind = var2ind[v1], var2ind[v2]

            for val1 in range(self.dom_size[v1]):
                for val2 in range(self.dom_size[v2]):
                    cost_ind = len(x)
                    cost_feat = c_embed + [data[val1][val2]]
                    x.append(cost_feat)

                    src.append(cost_ind)
                    dest.append(func_ind)
                    # src.append(func_ind)
                    # dest.append(cost_ind)

                    src.append(v1_ind+val1)
                    dest.append(cost_ind)
                    src.append(cost_ind)
                    dest.append(v1_ind+val1)

                    src.append(v2_ind+val2)
                    dest.append(cost_ind)
                    src.append(cost_ind)
                    dest.append(v2_ind+val2)
        self.x = x
        return torch.tensor(x, dtype=torch.float32, device=self.device), \
               torch.tensor(edge_index, dtype=torch.long, device=self.device), all_func_ind, assign_len
