import os

import torch
from torch_geometric.data import HeteroData

from lagrtools.funcs import load_merged_graph

def collect_files(root):
    result = []

    for fname in os.listdir(root):
        if not fname.endswith('.npz'):
            continue

        path = os.path.join(root, fname)

        if not os.path.isfile(path):
            continue

        result.append(path)

    result.sort()
    return result

def convert_merged_graph(merged_graph):
    result = HeteroData()

    for (k, v) in merged_graph.items():
        if k[0] == 'node':
            io   = k[1]
            name = k[2]

            if io == 'x':
                result[name].x = torch.from_numpy(v).float()
            elif io == 'y':
                if v.shape[1] > 0:
                    result[name].y = torch.from_numpy(v).float()
            else:
                raise ValueError(f'Unknown io type: {io}')

        elif k[0] == 'edge':
            edge_triplet = (k[1], f'{k[1]}-{k[2]}', k[2])
            result[edge_triplet].edge_index = torch.from_numpy(v.T).long()

    return result

class LAGRDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform = None):
        self._root      = root
        self._transform = transform
        self._files     = collect_files(self._root)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        path         = self._files[index]
        merged_graph = load_merged_graph(path)
        graph        = convert_merged_graph(merged_graph)

        if self._transform is not None:
            graph = self._transform(graph)

        return graph

