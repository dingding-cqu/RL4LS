import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_sum, scatter_mean


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        cnt_vector_len = 16
        self.conv1 = GATConv(in_channels, 8, heads=8)
        self.conv2 = GATConv(64, 8, heads=8)
        self.conv3 = GATConv(64, 8, heads=8)
        self.conv4 = GATConv(64, out_channels, heads=4, concat=False)
        self.pooling_proj = nn.Linear(out_channels, out_channels, bias=False)
        self.target_proj = nn.Linear(out_channels, out_channels, bias=False)
        self.out = nn.Linear(out_channels * 2 + cnt_vector_len, 1)

    def forward(self, batch, decision_var_idxes):
        function_idxes = []
        s = 0
        flag = []
        for i in range(batch.num_graphs):
            data = batch.get_example(i)
            function_idxes += [j + s for j in data.function_idx]
            flag += [i] * len(data.function_idx)
            decision_var_idxes[i] += s
            s += data.x.shape[0]
        flag = torch.tensor(flag, device=batch.x.device)

        x = self.conv1(batch.x, batch.edge_index)
        x = F.elu(x)
        x = self.conv2(x, batch.edge_index)
        x = F.elu(x)
        x = self.conv3(x, batch.edge_index)
        x = F.elu(x)
        x = F.elu(self.conv4(x, batch.edge_index))

        function_pooling = x[function_idxes]
        function_pooling = scatter_mean(function_pooling, flag, dim=0)
        function_pooling = self.pooling_proj(function_pooling)
        target = x[decision_var_idxes, :]
        target = self.target_proj(target)
        cnt_vector = torch.tensor(batch.cnt_vector, dtype=torch.float32, device=x.device)
        return self.out(F.elu(torch.cat([function_pooling, target, cnt_vector], dim=1)))

    @torch.no_grad()
    def inference(self, x, edge_index, cnt_vector, domain_size, function_idx):
        decision_var_idxes = [i for i in range(domain_size)]
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.elu(self.conv4(x, edge_index))

        function_pooling = x[function_idx].mean(dim=0)
        function_pooling = self.pooling_proj(function_pooling)

        function_pooling = function_pooling.repeat((len(decision_var_idxes), 1))
        cnt_vector = torch.tensor(cnt_vector, dtype=torch.float32, device=x.device)
        cnt_vector = cnt_vector.repeat((len(decision_var_idxes), 1))
        target = x[decision_var_idxes, :]
        target = self.target_proj(target)

        return self.out(F.elu(torch.cat([function_pooling, target, cnt_vector], dim=1)))
