import json
import os
from typing import Any, Dict, List, Tuple

from torch_geometric.transforms.base_transform import BaseTransform
import numpy as np

# FeaturePath : (node_name, io_type)
FeaturePath = Tuple[str, str]

# StatDict : { feature_path : { stat : [ value, ] } }
StatDict    = Dict[FeaturePath, Dict[str, List[float]]]

def unpack_stat_name(name : str) -> FeaturePath:
    tokens = name.split(':', maxsplit = 2)
    assert tokens[0] == 'node'

    io_type   = tokens[1]
    node_name = tokens[2]

    return (node_name, io_type)

class NodeFeatureNorm(BaseTransform):

    def __init__(
        self,
        root      : str,
        norm_type : str   = 'standartize',
        eps       : float = 1e-6
    ):
        self._root       = root
        self._stats_dict = NodeFeatureNorm.load_feature_stats(root)
        self._norm_type  = norm_type
        self._eps        = eps

    @staticmethod
    def load_feature_stats(root : str) -> StatDict:
        path = os.path.join(root, 'stats.json')

        with open(path, 'rt', encoding = 'utf-8') as f:
            result = json.load(f)

        return {
            unpack_stat_name(name) : {
                stat : np.array(values, dtype = np.float32)
                for (stat, values) in values_dict.items()
            }
            for (name, values_dict) in result.items()
        }

    def normalize(self, data, node, io_type):
        feature_path = (node, io_type)
        if feature_path not in self._stats_dict:
            return

        if self._norm_type == 'standartize':
            mean = self._stats_dict[feature_path]['mean']
            std  = self._stats_dict[feature_path]['stdev']

            data[node][io_type] -= mean
            data[node][io_type] /= (std + self._eps)

        else:
            raise ValueError(f'Unknown norm type: {self._norm_type}')

    def __call__(self, data: Any) -> Any:
        for node_name in data.node_types:
            for io_type in data[node_name].keys():
                self.normalize(data, node_name, io_type)

        return data

