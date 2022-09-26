from typing import Dict, Set, Tuple, List, Optional
import numpy as np

NODE_FEATURES = {
    'cnodes' : [ 'ident', 'value', 'uncertainty', 'index', 'wpid' ],
    'wnodes' : [
        'ident', 'wip', 'segment', 'channel', 'plane',
        'tailx', 'taily', 'tailz', 'headx', 'heady', 'headz'
    ],
    'bnodes' : [
        'ident', 'value', 'uncertainty', 'faceid', 'sliceid', 'start', 'span',
        'min1', 'max1', 'min2', 'max2', 'min3', 'max3',
        'ncorners', 'corner_coords'
    ],
    'snodes' : [ 'ident', 'value', 'uncertainty', 'frameid', 'start', 'span' ],
    'mnodes' : [ 'ident', 'value', 'uncertainty', 'wpid' ],
}

NODE_IDS = {
    'cnodes' : [ 'ident', ],
    'wnodes' : [ 'ident',  'channel', ],
    'bnodes' : [ 'ident', ],
    'snodes' : [ 'ident', ],
    'mnodes' : [ 'ident', ],
}

FeatureConfig = Dict[str, Optional[List[str]]]
NodeId        = Tuple[int, ...]

NODE_FEATURE_IDX_MAP = {
    node : {
        feature : idx for (idx, feature) in enumerate(features)
   } for (node, features) in NODE_FEATURES.items()
}

class Nodes:

    def __init__(self, ids : np.ndarray, values : np.ndarray) -> None:
        self._ids    = ids
        self._values = values
        self._id_set = set(tuple(x) for x in ids)
        self._id_index_map = { tuple(x) : i for (i, x) in enumerate(ids) }

        assert len(self._ids) == len(self._id_set), \
            'Duplicated ids found'

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def id_set(self) -> Set[NodeId]:
        return self._id_set

    @property
    def id_index_map(self) -> Dict[NodeId, int]:
        return self._id_index_map

    @property
    def values(self) -> np.ndarray:
        return self._values

    def __len__(self):
        return len(self._ids)

    def __repr__(self):
        return f'Nodes({len(self)})'

    def __eq__(self, other):
        if not np.all(self._ids == other._ids):
            return False

        if not np.all(np.isclose(self._values, other._values)):
            return False

        return True

    @staticmethod
    def get_ids(name : str, values : np.ndarray) -> np.ndarray:
        id_indices  = [
            NODE_FEATURE_IDX_MAP[name][feature] for feature in NODE_IDS[name]
        ]
        result = values[:, id_indices]

        return result.astype(np.int32)

    @staticmethod
    def from_values(
        name : str, values : np.ndarray,
        feature_config : Optional[FeatureConfig]
    ) -> Optional['Nodes']:

        if (feature_config is not None) and (name not in feature_config):
            return None

        ids               = Nodes.get_ids(name, values)
        feature_config    = feature_config or {}
        selected_features = feature_config.get(name, None)

        if selected_features is not None:
            selected_indices  = [
                NODE_FEATURE_IDX_MAP[name][feature]
                    for feature in selected_features
            ]
            values = values[:, selected_indices]

        return Nodes(ids, values)

    def ids_in_set_mask(self, filter_ids : Set[NodeId]) -> np.ndarray:
        mask = np.isin(self.ids, list(filter_ids), assume_unique = True)
        mask = np.all(mask, axis = 1)
        return mask

    def filter(self, mask : np.ndarray) -> 'Nodes':
        if mask is None:
            return self

        return Nodes(self.ids[mask], self.values[mask])

    def __getstate__(self):
        return { 'ids': self._ids, 'values' : self._values, }

    def __setstate__(self, state_dict):
        return Nodes(state_dict['ids'], state_dict['values'])
