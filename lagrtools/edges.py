from typing import Set, Optional
import numpy as np

class Edges:

    def __init__(self, values : np.ndarray):
        self._values  = values
        self._src_set = set(values[:, 0])
        self._dst_set = set(values[:, 1])

    @property
    def src_set(self) -> Set[int]:
        return self._src_set

    @property
    def dst_set(self) -> Set[int]:
        return self._dst_set

    @property
    def values(self) -> np.ndarray:
        return self._values

    def __len__(self):
        return len(self._values)

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        return np.all(self.values == other.values)

    def __repr__(self):
        return f'Edges({len(self)})'

    def filter(
        self,
        node_mask_src : Optional[np.ndarray],
        node_mask_dst : Optional[np.ndarray]
    ) -> 'Edges':

        reindex_src = reindex_map(node_mask_src)
        reindex_dst = reindex_map(node_mask_dst)

        new_edges = self.values.copy()

        if reindex_src is not None:
            new_edges[:, 0] = reindex_src[new_edges[:, 0]]

        if reindex_dst is not None:
            new_edges[:, 1] = reindex_dst[new_edges[:, 1]]

        drop_mask = np.any(new_edges == -1, axis = 1)

        return Edges(new_edges[~drop_mask])

    def __getstate__(self):
        return { 'values' : self._values, }

    def __setstate__(self, state_dict):
        return Edges(state_dict['values'])

def reindex_map(node_mask : Optional[np.ndarray]) -> Optional[np.ndarray]:
    if node_mask is None:
        return None

    n_total = len(node_mask)
    n_keep  = np.count_nonzero(node_mask)

    result            = -np.ones(n_total, dtype = np.int32)
    result[node_mask] = np.arange(0, n_keep, step = 1, dtype = np.int32)

    return result

