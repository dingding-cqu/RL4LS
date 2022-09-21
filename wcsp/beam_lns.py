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
    def __init__(self, pth, model, device='cpu', p=.2, scale=10):
        self.env = LNSEnv(*parse(pth, scale=scale), device=device)
        self.model = model
        # self.p = p
        # self.assignment = {x: random.randrange(self.env.dom_size[x]) for x in self.env.dom_size.keys()}
        self.device = device
        self.scale = scale

    def step(self, init_assignment, p):
        assignment = random.sample(list(self.env.dom_size.keys()), int((1 - p) * len(self.env.dom_size)))
        assignment = {x: init_assignment[x] for x in assignment}
        self.env.set_assignments(assignment)
        new_assignment = dict(init_assignment)
        for i in range(len(self.env.dfs_tree)):
            root = self.env.root[i]
            if len(self.env.dfs_tree[i]) == 1:
                vec = [sum([self.env.function_table[root][n][v][init_assignment[n]] for n in self.env.adj_list[root]]) for v in range(self.env.dom_size[root])]
                new_assignment[root] = argmin(vec)
                continue
            self._dfs_decision_making(i, root, new_assignment)
        return new_assignment

    def total_cost(self, assignment):
        cost = 0
        for v1 in self.env.function_table:
            for v2, func in self.env.function_table[v1].items():
                cost += func[assignment[v1]][assignment[v2]]
        return int(cost * self.scale / 2)

    def _dfs_decision_making(self, i, target_var, new_assignment):
        if len(self.env.dfs_tree[i][target_var].children) == 0:
            vec = [sum([self.env.function_table[target_var][n][v][new_assignment[n]] for n in self.env.adj_list[target_var]]) for v
                   in range(self.env.dom_size[target_var])]
            new_assignment[target_var] = argmin(vec)
            return
        sep_assignment = {i: new_assignment[i] for i in self.env.dfs_tree[i][target_var].sep}
        x, edge_index, all_function_index, action_space = self.env.build_graph(tree_idx=i, target_var=target_var, partial_assignment=sep_assignment)
        q_values = self.model.inference(x.to(self.device), edge_index.to(self.device), action_space, all_function_index)
        action = q_values.argmax().item()
        new_assignment[target_var] = action
        for c in self.env.dfs_tree[i][target_var].children:
            self._dfs_decision_making(i, c, new_assignment)

    @classmethod
    def top_k(cls, data, k):
        sorted_data = sorted(enumerate(data), key=lambda x: x[1])
        idx = [i[0] for i in sorted_data]
        return idx[:k]

    def beam_lns(self, max_cycle, k_b, k_ext, res_pth):
        B = []
        for i in range(k_b):
            sol = {x: random.randrange(self.env.dom_size[x]) for x in self.env.dom_size.keys()}
            B.append(sol)
        best_cost = 10**8
        cur_cycle = 0
        cur_t = perf_counter()
        f_p = open(res_pth, 'w')
        while cur_cycle < max_cycle:
            B_prime = []
            for i in range(k_b):
                sol = B[i]
                for j in range(k_ext):
                    p = 0.2
                    new_assignment = self.step(sol, p)
                    B_prime.append(new_assignment)
                # B_prime.append(sol)
            costs = [0]*len(B_prime)
            for i in range(len(B_prime)):
                costs[i] = self.total_cost(B_prime[i])
            idx = LNSHeuristic.top_k(costs, k_b)
            best_cost = min(best_cost, costs[idx[0]])
            f_p.writelines(f'{cur_cycle}\t {(perf_counter() - cur_t):.2f} \t{costs[idx[0]]:.2f}\t{best_cost:.2f}\n')
            print(cur_cycle, '\t', costs[idx[0]], best_cost)
            for i in range(k_b):
                B[i] = dict(B_prime[idx[i]])
            cur_cycle += 1#k_b*k_ext
        f_p.close()
        return best_cost

if __name__ == '__main__':
    valid_files = []
    path =  r'../problems/problems4paper/70/0.1'
    for (dirpath, dirnames,files) in os.walk(path):
        for filename in files:
            if '.xml' in filename:
                filepath = os.path.join(dirpath, filename)
                valid_files.append(filepath)

    d = 'cpu'
    model_pth = '../models/292.pth'
    m = GATNet(4, 16)
    m.load_state_dict(torch.load(model_pth, map_location=d))
    dirs = r'../problems/problems4paper/70/0.1'
    # bc = []
    # for k_b in range(1,10,2):
    #     for k_ext in range(1,6,1):
    k_b,k_ext = 4,2
    print(k_b, ' ', k_ext)
    problem = valid_files[0]
    lns = LNSHeuristic(problem, m, d)
    res_pth = problem[:-4]+f'_model_beam_lns_{k_b}_{k_ext}.txt'
    best_cost = lns.beam_lns(max_cycle=1000,k_b=k_b, k_ext=k_ext, res_pth=res_pth)
    print(problem, best_cost, datetime.datetime.now())
