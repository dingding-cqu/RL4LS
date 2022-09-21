import os

from model import GATNet
from dqn_agent import DQNAgent
from torch.optim import AdamW


if __name__ == '__main__':
    train_pth = '../problems/train'
    valid_pth = '../problems/valid'

    train_files, valid_files = [], []

    for f in os.listdir(train_pth):
        if f.endswith('.xml'):
            train_files.append(os.path.join(train_pth, f))
    for f in os.listdir(valid_pth):
        if f.endswith('.xml'):
            valid_files.append(os.path.join(valid_pth, f))

    model = GATNet(4, 16)
    target_model = GATNet(4, 16)
    optimizer = AdamW(model.parameters(), lr=.0001, weight_decay=5e-5)
    dqn = DQNAgent(model, target_model, optimizer, device='cuda:0', capacity=1000000)
    dqn.train(train_files, valid_files)