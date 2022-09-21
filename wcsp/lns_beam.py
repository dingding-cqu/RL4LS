import datetime
import os
import random
from collections import namedtuple
from time import perf_counter
import torch
from numpy import argmin
from core.parser import parse
from core.utility import transpose
from model import GATNet

Node = namedtuple('Node', ['name', 'parent', 'all_parents', 'children', 'level', 'sep'])
x_embed = [1, 0, 0, 0]
c_embed = [0, 1, 0]
f_embed = [0, 0, 0, 1]
PA = namedtuple('PA', ['assign', 'cpa_cost'])
PA_ext = namedtuple('PA', ['assign', 'cpa_cost', 'eval_cost'])

class LNSEnv:
    def __init__(self, all_vars, all_functions, device='cpu'):
        self.dom_size = dict()
        for name, dom in all_vars:
            self.dom_size[name] = dom
        self.dfs_tree = []
        adj_list = dict()
        self.function_table = dict()
        for data, v1, v2 in all_functions:
            if v1 not in adj_list:
                adj_list[v1] = list()
                self.function_table[v1] = dict()
            if v2 not in adj_list:
                adj_list[v2] = list()
                self.function_table[v2] = dict()
            self.function_table[v1][v2] = data
            self.function_table[v2][v1] = transpose(data)
            adj_list[v1].append(v2)
            adj_list[v2].append(v1)
        self.root = []
        self.device = device
        self.assignment = dict()
        self.destroyed_variables = list()
        self.adj_list = adj_list

    def set_assignments(self, assignment):
        self.assignment = dict(assignment)
        self.destroyed_variables = [x for x in self.dom_size.keys() if x not in assignment]
        destroyed_variables = list(self.destroyed_variables)
        self.root.clear()
        self.dfs_tree.clear()
        while len(destroyed_variables) > 0:
            self._dfs(destroyed_variables)

    def _dfs(self, destroyed_variables, level=0, cur_node=None):
        if cur_node is None:
            cur_node = random.choice(destroyed_variables)
            self.root.append(cur_node)
            self.dfs_tree.append(dict())
            parent = None
            all_parents = set()
            sep = set()
        else:
            all_parents = set([x for x in self.adj_list[cur_node] if x in self.dfs_tree[-1]])
            sep = set(all_parents)
            parent = [x for x in all_parents if self.dfs_tree[-1][x].level == level - 1]
            assert len(parent) == 1
            parent = parent[0]
        self.dfs_tree[-1][cur_node] = Node(cur_node, parent, all_parents, set(), level, sep)
        for n in self.adj_list[cur_node]:
            if n not in self.dfs_tree[-1] and n in destroyed_variables:
                self.dfs_tree[-1][cur_node].children.add(n)
                self._dfs(destroyed_variables, level + 1, n)

        for n in self.dfs_tree[-1][cur_node].children:
            self.dfs_tree[-1][cur_node].sep.update(self.dfs_tree[-1][n].sep)
        self.dfs_tree[-1][cur_node].sep.discard(cur_node)
        destroyed_variables.remove(cur_node)

    def build_graph(self, tree_idx, partial_assignment, target_var):
        dfs_tree = self.dfs_tree[tree_idx]
        checksum = sum([0 if sep in partial_assignment else 1 for sep in dfs_tree[target_var].sep])
        assert checksum == 0
        x = []
        edge_index = [[], []]
        node_index = dict()
        all_function_node_index = []
        partial_assignment = dict(partial_assignment)
        partial_assignment.update(self.assignment)
        self._dfs_build_graph(dfs_tree, partial_assignment, target_var, x, edge_index, node_index, all_function_node_index)
        return torch.tensor(x, dtype=torch.float32, device=self.device), \
               torch.tensor(edge_index, dtype=torch.long, device=self.device), all_function_node_index, \
               [node_index[target_var] + i for i in range(self.dom_size[target_var])]

    def _dfs_build_graph(self, dfs_tree, partial_assignment, cur_var, x, edge_index, node_index, all_function_node_index):
        node_index[cur_var] = len(x)
        src, dest = edge_index
        for val in range(self.dom_size[cur_var]):
            x.append(x_embed)
        for p in self.adj_list[cur_var]:
            if p in partial_assignment and p not in dfs_tree[cur_var].all_parents:
                assert p not in dfs_tree
                f_idx = len(x)
                all_function_node_index.append(f_idx)
                x.append(f_embed)
                for val in range(self.dom_size[cur_var]):
                    idx = len(x)
                    x.append(c_embed + [self.function_table[cur_var][p][val][partial_assignment[p]]])
                    src.append(idx)
                    dest.append(f_idx)

                    src.append(idx)
                    dest.append(node_index[cur_var] + val)

        for p in dfs_tree[cur_var].all_parents:
            f_idx = len(x)
            all_function_node_index.append(f_idx)
            x.append(f_embed)
            if p in partial_assignment:
                for val in range(self.dom_size[cur_var]):
                    idx = len(x)
                    x.append(c_embed + [self.function_table[cur_var][p][val][partial_assignment[p]]])
                    src.append(idx)
                    dest.append(f_idx)

                    src.append(idx)
                    dest.append(node_index[cur_var] + val)
            else:
                for val_p in range(self.dom_size[p]):
                    for val in range(self.dom_size[cur_var]):
                        idx = len(x)
                        x.append(c_embed + [self.function_table[cur_var][p][val][val_p]])
                        src.append(idx)
                        dest.append(f_idx)

                        src.append(idx)
                        dest.append(node_index[p] + val_p)

                        src.append(node_index[cur_var] + val)
                        dest.append(idx)
        for c in dfs_tree[cur_var].children:
            self._dfs_build_graph(dfs_tree, partial_assignment, c, x, edge_index, node_index, all_function_node_index)


class LNSHeuristic:
    def __init__(self, pth, model, k_ext, k_bw, device='cpu', p=.2, scale=10):
        self.env = LNSEnv(*parse(pth, scale=scale), device=device)
        self.model = model
        self.p = p
        self.assignment = {x: random.randrange(self.env.dom_size[x]) for x in self.env.dom_size.keys()}
        self.device = device
        self.scale = scale
        self.k_ext = k_ext
        self.k_bw = k_bw

    def step(self):
        assignment = random.sample(list(self.env.dom_size.keys()), int((1 - self.p) * len(self.env.dom_size)))
        assignment = {x: self.assignment[x] for x in assignment}
        self.env.set_assignments(assignment)
        # new_assignment = dict(self.assignment)
        for i in range(len(self.env.dfs_tree)):
            root = self.env.root[i]
            if len(self.env.dfs_tree[i]) == 1:
                vec = [sum([self.env.function_table[root][n][v][self.assignment[n]] for n in self.env.adj_list[root]]) for v in range(self.env.dom_size[root])]
                self.assignment[root] = argmin(vec)
                continue

            self.B = dict()
            for i in range(k_bw):
                self.B[i] = PA({}, 0)
            level_var = dict()
            for var in self.env.dfs_tree[i]:
                level = self.env.dfs_tree[i][var].level
                if level not in level_var:
                    level_var[level] = []
                level_var[level].append(var)

            for level in range(max(level_var.keys())+1):
                for target_var in level_var[level]:
                    self._decision_making(i, target_var)
            costs = [self.B[i].cpa_cost for i in range(len(self.B))]
            best_idx = argmin(costs)
            self.assignment.update(self.B[best_idx].assign)
        return self.total_cost(self.assignment)

    def total_cost(self, assignment):
        cost = 0
        for v1 in self.env.function_table:
            for v2, func in self.env.function_table[v1].items():
                cost += func[assignment[v1]][assignment[v2]]
        return int(cost * self.scale / 2)

    def _local_cost(self, tree_id, target_var, extend_assign):
        r = sum([self.env.function_table[target_var][p][extend_assign[target_var]][extend_assign[p]] for p in self.env.dfs_tree[tree_id][target_var].all_parents]) + \
            sum([self.env.function_table[target_var][n][extend_assign[target_var]][self.assignment[n]] for n in self.env.adj_list[target_var] if n not in self.env.destroyed_variables])
        return r

    def _decision_making(self, tree_id, target_var):
        B_prime = dict()
        for i in range(self.k_bw):
            assign = self.B[i].assign
            # if len(self.env.dfs_tree[tree_id][target_var].children) == 0:
            #     vec = [0] * self.env.dom_size[target_var]
            #     for val in range(self.env.dom_size[target_var]):
            #         assign[target_var] = val
            #         vec[val] = self._local_cost(tree_id, target_var, assign)
            #     assign[target_var] = argmin(vec)
            #     continue
            sep_assign = {key : assign[key] for key in self.env.dfs_tree[tree_id][target_var].sep}
            x, edge_index, all_function_index, action_space = self.env.build_graph(tree_idx=tree_id, target_var=target_var, partial_assignment=sep_assign)
            q_values = - self.model.inference(x.to(self.device), edge_index.to(self.device), action_space, all_function_index)
            q_values = [x for j in q_values.numpy().tolist() for x in j]
            actions = self.top_k(q_values, min(self.k_ext, len(q_values)))
            for j in range(len(actions)):
                extend_assign = dict(assign)
                extend_assign[target_var] = actions[j]
                r = self._local_cost(tree_id, target_var, extend_assign)
                B_prime[i*(len(actions))+j] = PA_ext(extend_assign, r + self.B[i].cpa_cost,
                                                     r + self.B[i].cpa_cost + q_values[actions[j]])
        costs = [B_prime[i].eval_cost for i in range(len(B_prime))]
        idx = self.top_k(costs, min(len(costs), self.k_bw))
        for i in range(len(idx)):
            self.B[i] = PA(dict(B_prime[idx[i]].assign), B_prime[idx[i]].cpa_cost)

    def _dfs_decision_making(self, i, target_var, partial_assignment, init_assignment):
        if len(self.env.dfs_tree[i][target_var].children) == 0:
            vec = [sum([self.env.function_table[target_var][n][v][init_assignment[n]] for n in self.env.adj_list[target_var]]) for v
                   in range(self.env.dom_size[target_var])]
            partial_assignment[target_var] = argmin(vec)
            return
        x, edge_index, all_function_index, action_space = self.env.build_graph(tree_idx=i, target_var=target_var, partial_assignment=partial_assignment)
        q_values = self.model.inference(x.to(self.device), edge_index.to(self.device), action_space, all_function_index)
        action = q_values.argmax().item()
        partial_assignment[target_var] = action
        for c in self.env.dfs_tree[i][target_var].children:
            self._dfs_decision_making(i, c, partial_assignment, init_assignment)

    @classmethod
    def top_k(cls, data, k):
        sorted_data = sorted(enumerate(data), key=lambda x: x[1])
        idx = [i[0] for i in sorted_data]
        return idx[:k]

if __name__ == '__main__':


    valid_files = []
    path =  r'../problems/problems4paper/var_vary/wgc_vary_vars'
    for (dirpath, dirnames,files) in os.walk(path):
        for filename in files:
            if '.xml' in filename:
                filepath = os.path.join(dirpath, filename)
                valid_files.append(filepath)

    model_pth = '../models/292.pth'
    m = GATNet(4, 16)
    m.load_state_dict(torch.load(model_pth, map_location='cpu'))
    for k_bw in range(3,10,2):
        for k_ext in range(1,6,1):
            print(k_bw, ' ', k_ext)
            # for i in range(len(valid_files)):
            problem = valid_files[1]
            lns = LNSHeuristic(problem, m, k_ext=k_ext, k_bw=k_bw)
            res_pth = problem[:-4]+f'_model_lns_beam_{k_bw}_{k_ext}.txt'
            best_cost = 10**8
            cur_cycle, max_cycle = 0, 1000
            cur_t = perf_counter()
            f_p = open(res_pth, 'w')
            while cur_cycle < max_cycle:
                cost = lns.step()
                best_cost = min(best_cost, cost)
                f_p.writelines(f'{cur_cycle}\t {(perf_counter() - cur_t):.2f} \t{cost:.2f}\t{best_cost:.2f}\n')
                # print(cur_cycle, '\t', best_cost)
                cur_cycle += 1
            f_p.close()
            print(problem, best_cost, datetime.datetime.now())
