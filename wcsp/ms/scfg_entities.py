from entities import FactorGraph


def _matrix_multiply(matrix, coefficient):
    res = []
    for row in matrix:
        res.append([x * coefficient for x in row])
    return res


class SCFG(FactorGraph):
    def __init__(self, pth, function_node_type, variable_node_type, split_ratio=.95):
        self.split_ratio = split_ratio
        super().__init__(pth, function_node_type, variable_node_type)

    def _construct_nodes(self, all_vars, all_matrix):
        for v, dom in all_vars:
            self.variable_nodes[v] = self.variable_node_type(v, dom)
        for matrix, row, col in all_matrix:
            matrix1 = _matrix_multiply(matrix, self.split_ratio)
            matrix2 = _matrix_multiply(matrix, 1 - self.split_ratio)
            self.function_nodes.append(self.function_node_type(f'({row},{col})1', matrix1, self.variable_nodes[row],
                                                    self.variable_nodes[col]))
            self.function_nodes.append(self.function_node_type(f'({row},{col})2', matrix2, self.variable_nodes[row],
                                                               self.variable_nodes[col]))