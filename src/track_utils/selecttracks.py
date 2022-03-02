
from typing import Set, Dict, List, Sequence

import napari
from napari.layers import Tracks, Labels, Layer
from napari.utils.progress import progress

from magicgui.widgets import Container, create_widget, PushButton

import numpy as np
import pandas as pd
import zarr

from track_utils.utils import zarr_key_to_slice, get_initialized_keys


class SelectTracks(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        self._tracks_layer = create_widget(annotation=Tracks, name='Tracks')
        self._tracks_layer.changed.connect(self._set_button_status)
        self.append(self._tracks_layer)

        self._segms_layer = create_widget(annotation=Labels, name='Segms (NodeIDs)')
        self._segms_layer.changed.connect(self._set_button_status)
        self.append(self._segms_layer)

        self._mask_layer = create_widget(annotation=Labels, name='Selection')
        self._mask_layer.changed.connect(self._set_button_status)
        self.append(self._mask_layer)

        self._select_button = PushButton(text='Select', enabled=self._button_status())
        self._select_button.changed.connect(self._on_select_clicked)
        self.append(self._select_button)
    
    def _on_select_clicked(self) -> None:
        track_layer: Tracks = self._tracks_layer.value
        features = track_layer.features
        data = track_layer.data
        graph = track_layer.graph
    
        df = pd.concat(
            (pd.DataFrame(data), features),
            axis=1,
        ).set_index('NodeID', drop=False)

        selected_track_ids = self._find_colliding_tracks(df)

        graph = {
            n: list(filter(lambda x: x in selected_track_ids, p))
            for n, p in graph.items()
            if n in selected_track_ids
        }

        data_len = data.shape[1]
        df = df.loc[df['track_id'].isin(selected_track_ids)]
        data = df.iloc[:, :data_len]
        features = df.iloc[:, data_len:]

        self._viewer.add_tracks(data=data, graph=graph, features=features, name='Selection')

    def _find_colliding_tracks(self, df: pd.DataFrame) -> np.ndarray:
        mask = self._mask_layer.value.data
        segms = self._segms_layer.value.data

        assert mask.shape == segms.shape

        selected_node_ids = set()
        if isinstance(mask, zarr.Array):
            for key in progress(list(get_initialized_keys(mask))):
                slicing = zarr_key_to_slice(mask, key)
                c_mask = mask[slicing]
                c_segms = segms[slicing]
                
                marked = np.unique(c_segms[c_mask > 0])
                for m in marked:
                    selected_node_ids.add(m)
        else:
            marked = np.unique(segms[mask > 0])
            for m in marked:
                selected_node_ids.add(m)
        
        if 0 in selected_node_ids:
            selected_node_ids.remove(0)
            
        selected_track_ids = df.loc[selected_node_ids, 'track_id'].unique()
        selected_track_ids = self._propagate_selection(selected_track_ids)
        return selected_track_ids
   
    def _propagate_selection(self, selected_track_ids: Sequence[int]) -> Set[int]:
        if not isinstance(selected_track_ids, Set):
            selected_track_ids = set(selected_track_ids)
        else:
            selected_track_ids = selected_track_ids.copy()

        orig_graph = self._tracks_layer.value.graph
        inv_graph = self._create_inv_graph(orig_graph)
        queue = list(selected_track_ids)

        while len(queue) > 0:
            track_id = queue.pop()

            neighbors = inv_graph.get(track_id, []) +\
                 orig_graph.get(track_id, [])

            for neigh in neighbors:
                if neigh not in selected_track_ids:
                    queue.append(neigh)
                    selected_track_ids.add(neigh)

        return selected_track_ids

    @staticmethod
    def _create_inv_graph(graph: Dict[int, List[int]]) -> Dict[int, List[int]]:
        inv_graph = {}
        for node, parents in graph.items():
            for parent in parents:
                if parent in inv_graph:
                    inv_graph[parent].append(node)
                else:
                    inv_graph[parent] = [node]
        return inv_graph

    def _set_button_status(self, layer: Layer) -> None:
        self._select_button.enabled = self._button_status()

    def _button_status(self) -> bool:
        return (
            self._tracks_layer.value is not None and 
            self._mask_layer.value is not None and
            self._segms_layer.value is not None
        )
