from typing import Dict, Tuple, Optional
import numpy as np

from .nodes import Nodes, FeatureConfig
from .edges import Edges

NodesDict = Dict[str, Nodes]
EdgesDict = Dict[Tuple[str, str], Edges]
MaskDict  = Dict[str, np.ndarray]

class Graph:

    def __init__(self, nodes : NodesDict, edges : EdgesDict):
        self._nodes = nodes
        self._edges = edges

    @property
    def nodes_dict(self) -> NodesDict:
        return self._nodes

    @property
    def edges_dict(self) -> EdgesDict:
        return self._edges

    def __repr__(self):
        result = 'Graph:\n'

        result += '  Nodes:\n'
        for key in sorted(self.nodes_dict.keys()):
            result += f'    {key}: ' + repr(self.nodes_dict[key]) + '\n'

        result += '  Edges:\n'
        for key in sorted(self.edges_dict.keys()):
            result += f'    {key}: ' + repr(self.edges_dict[key]) + '\n'

        return result

    def __eq__(self, other):
        return (
                (self.nodes_dict == other.nodes_dict)
            and (self.edges_dict == other.edges_dict)
        )

    def filter(self, mask_dict : MaskDict) -> 'Graph':
        new_nodes_dict : NodesDict = {}
        new_edges_dict : EdgesDict = {}

        for (k, nodes) in self.nodes_dict.items():
            new_nodes_dict[k] = nodes.filter(mask_dict.get(k, None))

        for (edge_name, edges) in self.edges_dict.items():
            k_src = edge_name[0]
            k_dst = edge_name[1]

            new_edges_dict[edge_name] = edges.filter(
                mask_dict.get(k_src, None), mask_dict.get(k_dst, None)
            )

        return Graph(new_nodes_dict, new_edges_dict)

def parse_edge_name(name : str) -> Tuple[str, str]:
    src = name[0]
    dst = name[1]

    return (src + 'nodes', dst + 'nodes')

def load_single_graph_from_dict(
    graph_dict     : Dict[str, np.ndarray],
    feature_config : Optional[FeatureConfig] = None
) -> Graph:
    nodes_dict = { }
    edges_dict = { }

    for name, values in graph_dict.items():
        if name.endswith('nodes'):
            nodes = Nodes.from_values(name, values, feature_config)

            if nodes is not None:
                nodes_dict[name] = nodes

    for name, values in graph_dict.items():
        if name.endswith('edges'):
            edge_tuple = parse_edge_name(name)

            if (
                   (edge_tuple[0] not in nodes_dict)
                or (edge_tuple[1] not in nodes_dict)
            ):
                continue

            edges_dict[edge_tuple] = Edges(values)

    return Graph(nodes_dict, edges_dict)

def load_single_graph(
    path : str, feature_config : Optional[FeatureConfig] = None
) -> Graph:
    with np.load(path) as f:
        return load_single_graph_from_dict(f, feature_config)

