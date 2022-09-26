from typing import Dict, Tuple
import toml
import numpy as np

from .graph import Graph
from .nodes import FeatureConfig

MergedGraph = Dict[Tuple[str,...], np.ndarray]

def parse_features_config(path : str) -> Tuple[FeatureConfig, FeatureConfig]:
    z = toml.load(path)

    feature_config_img = z['img']
    feature_config_tru = z['tru']

    return feature_config_img, feature_config_tru

def construct_merged_graph(g_img : Graph, g_tru : Graph) -> MergedGraph:
    result : MergedGraph = {}

    for (name, img_nodes) in g_img.nodes_dict.items():
        result[('node', 'x', name)] = img_nodes.values

    for (edge_name, img_edges) in g_img.edges_dict.items():
        result[('edge', *edge_name)] = img_edges.values

    for (name, tru_nodes) in g_tru.nodes_dict.items():
        if name not in g_img.nodes_dict:
            continue

        img_nodes = g_img.nodes_dict[name]

        assert np.all(img_nodes.ids == tru_nodes.ids)

        result[('node', 'y', name)] = tru_nodes.values

    return result

def save_merged_graph(path : str, graph : MergedGraph) -> None:
    def flatten_key(key : Tuple[str, ...]) -> str:
        return ':'.join(key)

    np.savez_compressed(
        path, **{ flatten_key(k) : v for (k, v) in graph.items() }
    )

def load_merged_graph(path : str) -> MergedGraph:
    result = {}

    with np.load(path) as f:
        for key in f:
            parsed_key = tuple(key.split(':', maxsplit = 2))
            result[parsed_key] = f[key]

    return result

