import os

from entities import VariableNode, FunctionNode
from scfg_entities import SCFG


if __name__ == '__main__':
    pth = r'C:\Users\ycdeng\Documents\WeChat Files\wxid_xgfmuy8pxdvf22\FileStorage\File\2022-01\code\code\problems'
    leap = ''
    VariableNode.damp_factor = .9
    for directory in ['valid_model']:
        cur_pth = f'{pth}/{directory}{"" if leap == "" else "/" + leap}'
        cic_pth = f'{cur_pth}/scfg_cic.txt'
        bcic_pth = f'{cur_pth}/scfg_bcic.txt'
        for rp in [cic_pth, bcic_pth]:
            if os.path.exists(rp):
                os.remove(rp)
        for pf in os.listdir(cur_pth):
            if not pf.endswith('.xml'):
                continue
            cic = []
            bcic = []
            fg = SCFG(f'{cur_pth}/{pf}', FunctionNode, VariableNode)
            for it in range(1000):
                cost = fg.step()
                cic.append(int(cost))
                if len(bcic) == 0:
                    bcic.append(cic[-1])
                else:
                    bcic.append(min(bcic[-1], cic[-1]))
            print(directory, pf, bcic[-1])
            with open(cic_pth, 'a') as rf:
                rf.write(str(cic) + '\n')
            with open(bcic_pth, 'a') as rf:
                rf.write(str(bcic) + '\n')