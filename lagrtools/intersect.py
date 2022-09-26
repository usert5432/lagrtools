from typing import Tuple

import numpy as np

from .nodes import Nodes
from .graph import NodesDict, MaskDict, Graph

def get_node_intersection_masks(
    n1 : Nodes, n2 : Nodes
) -> Tuple[np.ndarray, np.ndarray]:
    common_ids = n1.id_set.intersection(n2.id_set)

    mask1 = n1.ids_in_set_mask(common_ids)
    mask2 = n2.ids_in_set_mask(common_ids)

    return (mask1, mask2)

def get_node_dict_intersection_mask_dicts(
    d1 : NodesDict, d2 : NodesDict
) -> Tuple[MaskDict, MaskDict]:
    mask_dict1 = { k : None for k in d1.keys() }
    mask_dict2 = { k : None for k in d2.keys() }

    k1 = set(d1.keys())
    k2 = set(d2.keys())

    keys_common = k1.intersection(k2)

    for key in keys_common:
        n1 = d1[key]
        n2 = d2[key]

        mask1, mask2 = get_node_intersection_masks(n1, n2)

        mask_dict1[key] = mask1
        mask_dict2[key] = mask2

    return (mask_dict1, mask_dict2)

def graph_intersection(g1 : Graph , g2 : Graph) -> Tuple[Graph, Graph]:
    mask_dict1, mask_dict2 = \
        get_node_dict_intersection_mask_dicts(g1.nodes_dict, g2.nodes_dict)

    return (g1.filter(mask_dict1), g2.filter(mask_dict2))

