import zarr
import numpy as np
import pandas as pd

from typing import Dict

import napari
from napari.layers import Tracks
from napari.layers.tracks._managers import InteractiveTrackManager
from napari.utils.progress import progress

from track_utils.utils import zarr_key_to_slice, get_initialized_keys
from track_utils.selecttracks import SelectTracks
from magicgui.widgets import LineEdit


class LabelTracks(SelectTracks):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(viewer=viewer)

        self._select_button.text = 'Label'
        self._feature_name = LineEdit(label='Feature Name:', value='labels')

        self.append(self._feature_name)
    
    @staticmethod
    def _set_dict_unique_value(dict: Dict, key: int, value: int, mixed_value: int = -1) -> bool:
        cur_value = dict.get(key)
        if cur_value is None:
            dict[key] = value
        elif cur_value != value and cur_value != mixed_value:
            dict[key] = mixed_value
        else:
            return False
        # returns true if value changed
        return True

    def _find_colliding_tracks(self, df: pd.DataFrame) -> np.ndarray:
        mask = self._mask_layer.value.data
        segms = self._segms_layer.value.data

        assert mask.shape == segms.shape

        node_labels = {}
        if isinstance(mask, zarr.Array):
            for key in progress(list(get_initialized_keys(mask))):
                slicing = zarr_key_to_slice(mask, key)
                c_mask = mask[slicing]
                c_segms = segms[slicing]
                
                coords = np.where(c_mask > 0)
                both = np.stack((c_segms[coords], c_mask[coords]), axis=-1)
                both = np.unique(both, axis=0)
                for row in both:
                    self._set_dict_unique_value(node_labels, row[0], row[1])
       
        else:
            raise NotImplementedError
        
        if 0 in node_labels:
            del node_labels[0]
            
        track_labels = {}
        for node_id, node_lb in node_labels.items():
            self._set_dict_unique_value(track_labels, node_id, node_lb)

        track_labels = self._propagate_selection(track_labels)
        return track_labels

    def _propagate_selection(self, track_labels: Dict[int, int]) -> Dict[int, int]:

        orig_graph = self._tracks_layer.value.graph
        inv_graph = self._create_inv_graph(orig_graph)
        queue = list(track_labels.keys())

        while len(queue) > 0:
            track_id = queue.pop()
            track_lb = track_labels[track_id]

            neighbors = inv_graph.get(track_id, []) +\
                 orig_graph.get(track_id, [])

            for neigh in neighbors:
                if self._set_dict_unique_value(track_labels, neigh, track_lb):
                    queue.append(neigh)
 
        return track_labels

    def _on_select_clicked(self) -> None:
        track_layer: Tracks = self._tracks_layer.value
        manager: InteractiveTrackManager = track_layer._manager

        df = track_layer.features
        df.set_index('NodeID', drop=False, inplace=True)

        selected_track_ids = self._find_colliding_tracks(df)

        name = self._feature_name.value
        for node_id, track_id in progress(zip(df['NodeID'], df['track_id'])):
            manager._get_node(node_id).features[name] = selected_track_ids.get(track_id, 0)

        manager._is_serialized = False
        track_layer.events.properties()
        track_layer.events.color_by()
