import os
import random
from collections import namedtuple

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
        self.p = p
        self.assignment = {x: random.randrange(self.env.dom_size[x]) for x in self.env.dom_size.keys()}
        self.device = device
        self.scale = scale

    def reset(self):
        self.assignment = {x: random.randrange(self.env.dom_size[x]) for x in self.env.dom_size.keys()}
        cost = 0
        for v1 in self.env.function_table:
            for v2, func in self.env.function_table[v1].items():
                cost += func[self.assignment[v1]][self.assignment[v2]]
        return int(cost * self.scale / 2)

    def step(self):
        assignment = random.sample(list(self.env.dom_size.keys()), int((1 - self.p) * len(self.env.dom_size)))
        assignment = {x: self.assignment[x] for x in assignment}
        self.env.set_assignments(assignment)
        pa = dict()
        for i in range(len(self.env.dfs_tree)):
            root = self.env.root[i]
            if len(self.env.dfs_tree[i]) == 1:
                vec = [sum([self.env.function_table[root][n][v][self.assignment[n]] for n in self.env.adj_list[root]]) for v in range(self.env.dom_size[root])]
                self.assignment[root] = argmin(vec)
                continue
            self._dfs_decision_making(i, root, pa)
        self.assignment.update(pa)
        cost = 0
        for v1 in self.env.function_table:
            for v2, func in self.env.function_table[v1].items():
                cost += func[self.assignment[v1]][self.assignment[v2]]
        return int(cost * self.scale / 2)

    def _dfs_decision_making(self, i, target_var, partial_assignment):
        if len(self.env.dfs_tree[i][target_var].children) == 0:
            vec = [sum([self.env.function_table[target_var][n][v][self.assignment[n] if n not in self.env.dfs_tree[i][target_var].all_parents else partial_assignment[n]] for n in self.env.adj_list[target_var]]) for v
                   in range(self.env.dom_size[target_var])]
            self.assignment[target_var] = argmin(vec)
            return
        x, edge_index, all_function_index, action_space = self.env.build_graph(tree_idx=i, target_var=target_var, partial_assignment=partial_assignment)
        q_values = self.model.inference(x.to(self.device), edge_index.to(self.device), action_space, all_function_index)
        action = q_values.argmax().item()
        partial_assignment[target_var] = action
        for c in self.env.dfs_tree[i][target_var].children:
            self._dfs_decision_making(i, c, partial_assignment)


class BeamHeuristic:
    def __init__(self, pth, model, beam_width=8, beam_ext=1, device='cpu', p=.2, scale=10):
        self.lns = LNSHeuristic(pth, model, device, p, scale)
        self.beams = []
        self.beam_ext = beam_ext
        self.beam_width = beam_width
        for _ in range(beam_width):
            cost = self.lns.reset()
            self.beams.append((dict(self.lns.assignment), cost))

    def step(self):
        beam_prime = []
        for beam, _ in self.beams:
            for _ in range(self.beam_ext):
                self.lns.assignment = dict(beam)
                self.lns.p = 0.1 + 0.4 * random.random()
                cost = self.lns.step()
                beam_prime.append((dict(self.lns.assignment), cost))
        beam_prime += self.beams
        beam_prime = sorted(beam_prime, key=lambda x: x[-1])
        unique_beams = []
        current_cost = -1
        for b, c in beam_prime:
            if c != current_cost:
                current_cost = c
                unique_beams.append([])
            unique_beams[-1].append((b, c))
        self.beams = []
        exit = False
        while not exit:
            for lst in unique_beams:
                if len(lst) == 0:
                    continue
                beam = lst.pop(0)
                self.beams.append(beam)
                exit = len(self.beams) == self.beam_width
                if exit:
                    break
        return beam_prime[0][-1]


if __name__ == '__main__':
    d = 'cpu'
    model_pth = '../models/292.pth'
    m = GATNet(4, 16)
    m.load_state_dict(torch.load(model_pth, map_location=d))
    dirs = r'../problems/problems4paper/70/0.1'
    bc = []
    for prob in os.listdir(dirs):
        if not prob.endswith('.xml'):
            continue
        problem = f'{dirs}/{prob}'
        lns = BeamHeuristic(problem, m, device=d)
        best_cost = -1
        if len(bc) != 0:
            mean = sum(bc) / len(bc)
        else:
            mean = 0
        for _ in range(500):
            c = lns.step()
            if best_cost == -1:
                best_cost = c
            best_cost = min(best_cost, c)
            print(best_cost, c, mean, sep='\t')
        print(prob, best_cost)
        bc.append(best_cost)
    print(bc)