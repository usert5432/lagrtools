from typing import Any
import torch_geometric.transforms as T

class ThresholdLabel(T.BaseTransform):

    def __init__(
        self, node_name, io_type, target_label,
        feature_index = 0, threshold = 0
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self._node_name     = node_name
        self._io_type       = io_type
        self._target_label  = target_label
        self._feature_index = feature_index
        self._threshold     = threshold

    def __call__(self, data : Any) -> Any:
        label = data[self._node_name][self._io_type][:, self._feature_index]
        label = (label > self._threshold).unsqueeze(1)

        data[self._node_name][self._target_label] = label.int()

        return data

