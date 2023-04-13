"""
Utilities to handle segmentation
"""


from typing import List

import numpy as np

__all__ = ["MapLabels"]


class MapLabels:
    """
    Utility to map the original segmentation labels in a new set, see all the original
    segmentation labels by means of ``list_seg_labels``. You have to provide a dictionary
    mapping list of labels into their new index. segmentation maps are considered sparse
    where 0 means empty, original segmentation labels not mapped are masked.
    """

    def __init__(
        self, all_labels: np.array | List[str], mapping: dict[str, tuple[set[str], int]]
    ):
        self.mapping = {k: (np.array(list(v[0])), v[1]) for k, v in mapping.items()}
        self.all_labels = (
            all_labels if isinstance(all_labels, np.ndarray) else np.array(all_labels)
        )

    def __call__(self, segmentation_map: np.array) -> np.array:
        mask = segmentation_map >= 0
        valids = segmentation_map[mask]
        output = np.zeros_like(valids) - 100
        for labels, idx in self.mapping.values():
            (labels_to_idx,) = np.in1d(self.all_labels, labels).nonzero()
            output[np.in1d(valids, labels_to_idx)] = idx

        output_map = np.zeros_like(segmentation_map) - 100
        output_map[mask] = output
        return output_map
