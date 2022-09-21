import datetime
import os
# from time import perf_counter
# from entities import FactorGraph, VariableNode, FunctionNode

# from utilities import elementwise_add, argmin
# from parser_wcsp import parse
# from wcsp.core.parse_real_world_problems import parse

import xml.etree.ElementTree as ET


def parse(path, scale=1):
    agents = {}
    root = ET.parse(path).getroot()
    ele_agents = root.find('agents')
    for ele_agent in ele_agents.findall('agent'):
        id = ele_agent.get('name')
        agents[id] = [id, 0]
    domains = {}
    ele_domains = root.find('domains')
    for ele_domain in ele_domains.findall('domain'):
        id = ele_domain.get('name')
        nb_values = ele_domain.get('nbValues')
        domains[id] = int(nb_values)

    ele_variables = root.find('variables')
    for ele_variable in ele_variables.findall('variable'):
        agent_id = ele_variable.get('agent')
        domain_id = ele_variable.get('domain')
        agents[agent_id][-1] = domains[domain_id]
    constraints = {}
    relations = {}
    ele_constraints = root.find('constraints')
    for ele_constraint in ele_constraints.findall('constraint'):
        id = ele_constraint.get('name')
        scope = ele_constraint.get('scope').split(' ')
        scope = ['A' + s[1: -2] for s in scope]
        reference = ele_constraint.get('reference')
        constraints[id] = scope
        relations[reference] = id

    ele_relations = root.find('relations')
    all_matrix = []
    for ele_relation in ele_relations.findall('relation'):
        id = ele_relation.get('name')
        content = ele_relation.text.split('|')
        first_constraint = []
        for tpl in content:
            cost, values = tpl.split(':')
            cost = float(cost) / scale
            values = [int(s) for s in values.split(' ')]
            while len(first_constraint) < values[0]:
                first_constraint.append([])
            row = first_constraint[values[0] - 1]
            while len(row) < values[1]:
                row.append(0)
            row[values[1] - 1] = cost
        pair = constraints[relations[id]]
        all_matrix.append((first_constraint, pair[0], pair[1]))
    all_vars = []
    for data in agents.values():
        all_vars.append(tuple(data))
    return all_vars, all_matrix


def elementwise_add(list1, list2):
    assert len(list1) == len(list2)
    return [x + y for x, y in zip(list1, list2)]


def argmin(arr):
    best_val = arr[0]
    best_idx = 0
    for idx, val in enumerate(arr):
        if val < best_val:
            best_val = val
            best_idx = idx
    return best_idx

class Node:
    def __init__(self, name):
        self.name = name
        self.incoming_msg = dict()

    def compute_msgs(self):
        pass

    def __repr__(self):
        return self.name


class VariableNode(Node):
    damp_factor = 0
    op = argmin

    def __init__(self, name, dom_size):
        super().__init__(name)
        self.dom_size = dom_size
        self.prev_sent = dict()
        self.val_idx = -1
        self.neighbors = dict()  # neighbor id: neighbor node

    def register_neighbor(self, neighbor):
        self.neighbors[neighbor.name] = neighbor

    def compute_msgs(self):
        for nei in self.neighbors:
            msg = [0] * self.dom_size
            for other_nei in self.neighbors:
                if other_nei == nei:
                    continue
                if other_nei not in self.incoming_msg:
                    continue
                msg = elementwise_add(msg, self.incoming_msg[other_nei])
            norm = min(msg)
            msg = [x - norm for x in msg]
            # damping & normalizing
            if nei in self.prev_sent and 0 < VariableNode.damp_factor < 1:
                prev = self.prev_sent[nei]
                msg = [(1 - VariableNode.damp_factor) * x + VariableNode.damp_factor * y for x, y in zip(msg, prev)]
                norm = min(msg)
                msg = [x - norm for x in msg]
            self.prev_sent[nei] = list(msg)
            # send the message to nei
            self.neighbors[nei].incoming_msg[self.name] = msg

    def make_decision(self):
        belief = [0] * self.dom_size
        for nei in self.neighbors:
            if nei in self.incoming_msg:
                belief = elementwise_add(belief, self.incoming_msg[nei])
        self.val_idx = VariableNode.op(belief)


class FunctionNode(Node):
    op = min

    def __init__(self, name, matirx, row_vn, col_vn):
        super().__init__(name)
        self.matrix = matirx  # we deal with binary factors
        self.row_vn = row_vn
        self.col_vn = col_vn
        self.row_vn.register_neighbor(self)
        self.col_vn.register_neighbor(self)

    def compute_msgs(self):
        for nei in [self.row_vn, self.col_vn]:
            msg = [0] * nei.dom_size
            if nei == self.row_vn:
                belief = [0] * self.col_vn.dom_size if self.col_vn.name not in self.incoming_msg else self.incoming_msg[
                    self.col_vn.name]
                for val in range(self.row_vn.dom_size):
                    utils = [x + y for x, y in zip(belief, self.matrix[val])]
                    msg[val] = FunctionNode.op(utils)
            else:
                belief = [0] * self.row_vn.dom_size if self.row_vn.name not in self.incoming_msg else self.incoming_msg[
                    self.row_vn.name]
                for val in range(self.col_vn.dom_size):
                    local_vec = [self.matrix[i][val] for i in range(self.row_vn.dom_size)]
                    utils = [x + y for x, y in zip(belief, local_vec)]
                    msg[val] = FunctionNode.op(utils)
            nei.incoming_msg[self.name] = msg


class FactorGraph:
    def __init__(self, pth, function_node_type, variable_node_type):
        self.variable_nodes = dict()
        self.function_nodes = []
        self.function_node_type = function_node_type
        self.variable_node_type = variable_node_type
        all_vars, all_matrix = parse(pth,1)
        self._construct_nodes(all_vars, all_matrix)

    def _construct_nodes(self, all_vars, all_matrix):
        for v, dom in all_vars:
            self.variable_nodes[v] = self.variable_node_type(v, dom)
        for matrix, row, col in all_matrix:
            self.function_nodes.append(self.function_node_type(f'({row},{col})', matrix, self.variable_nodes[row],
                                                               self.variable_nodes[col]))
        all_degree = sum([len(x.neighbors) for x in self.variable_nodes.values()])
        # assert int(all_degree / 2) == len(self.function_nodes)

    def step(self):
        for func in self.function_nodes:
            func.compute_msgs()
        for variable in self.variable_nodes.values():
            variable.compute_msgs()
            variable.make_decision()
        cost = 0
        for func in self.function_nodes:
            cost += func.matrix[func.row_vn.val_idx][func.col_vn.val_idx]
        assign = {}
        for var in self.variable_nodes.values():
            assign[str(var)] = var.val_idx
        return cost, assign

def run_problem(pth, cycle=500, damped_factor=.9):
    # res_pth = pth[:-4]+'_dms.txt'
    # cur_t = perf_counter()
    # f_p = open(res_pth, 'w')
    VariableNode.damp_factor = damped_factor
    fg = FactorGraph(pth, FunctionNode, VariableNode)
    cost = []
    best_cost = []
    best_assign = {}
    best_assign_cost = 100000
    for it in range(cycle):
        cst, assign = fg.step()
        cost.append(cst)
        if best_assign_cost > cst:
            best_assign_cost = cst
            best_assign = assign

        if len(best_cost) == 0:
            best_cost.append(cost[-1])
        else:
            best_cost.append(min(best_cost[-1], cost[-1]))
    # assert best_assign_cost == best_cost[-1]
    return best_assign, best_assign_cost
    #     f_p.writelines(f'{it}\t {(perf_counter() - cur_t):.2f} \t{cost[-1]:.2f}\t{best_cost[-1]:.2f}  \n')
    #     # print(f'{it}\t {(perf_counter() - cur_t):.2f} \t{cost[-1]:.2f}\t{best_cost[-1]:.2f} ')
    # f_p.close()
    # print(pth, best_cost[-1], datetime.datetime.now())


# def run(problem_dir, cycle=1000, damped_factor=.9):
#     cic = []
#     bcic = []
#     cnt = 0
#     for f in os.listdir(problem_dir):
#         if not f.endswith('.xml'):
#             continue
#
#         cnt += 1
#         pth = os.path.join(problem_dir, f)
#         res_pth = pth[:-4]+'_dms.txt'
#         cur_t = perf_counter()
#         f_p = open(res_pth, 'w')
#         VariableNode.damp_factor = damped_factor
#         fg = FactorGraph(pth, FunctionNode, VariableNode)
#         cost = []
#         best_cost = []
#         for it in range(cycle):
#             cost.append(fg.step())
#
#             if len(best_cost) == 0:
#                 best_cost.append(cost[-1])
#             else:
#                 best_cost.append(min(best_cost[-1], cost[-1]))
#             f_p.writelines(f'{it}\t {(perf_counter() - cur_t):.2f} \t{cost[-1]:.2f}\t{best_cost[-1]:.2f}  \n')
#             print(f'{it}\t {(perf_counter() - cur_t):.2f} \t{cost[-1]:.2f}\t{best_cost[-1]:.2f}  \n')
#         f_p.close()
#         print(pth, best_cost[-1], datetime.datetime.now())
#         if len(cic) == 0:
#             cic = cost
#             bcic = best_cost
#         else:
#             cic = [x + y for x, y in zip(cost, cic)]
#             bcic = [x + y for x, y in zip(best_cost, bcic)]
#     return [x / cnt for x in cic], [x / cnt for x in bcic]
#
#
# if __name__ == '__main__':
#     # pth = r'../../problems/problems4paper/70/0.6'
#     # c, bc = run(pth)
#
#     valid_files = []
#     path = '../../problems/real_world_problems/celar-32'
#     for (dirpath, dirnames,files) in os.walk(path):
#         for filename in files:
#             if '.wcsp' in filename:
#                 filepath = os.path.join(dirpath, filename)
#                 valid_files.append(filepath)
#     valid_files = valid_files[12:]
#     for j in range(len(valid_files)):
#         run_problem(valid_files[j])

