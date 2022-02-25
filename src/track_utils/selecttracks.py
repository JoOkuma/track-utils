
from typing import Set

import napari
from napari.layers import Tracks, Labels, Layer
from napari.utils.progress import progress
from napari.layers.tracks._managers import InteractiveTrackManager

from magicgui.widgets import Container, create_widget, PushButton
import pandas as pd
from typing import Dict
import dask.array as da


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
        track_layer = self._tracks_layer.value
        manager: InteractiveTrackManager = track_layer._manager

        track_data = self._tracks_layer.value.data
        mask = self._mask_layer.value.data

        columns = ['TrackID', 't', 'z', 'y', 'x']
        if track_data.ndim == 4:
            columns.remove('z')

        df = pd.DataFrame(track_data, columns=columns)
        selected_track_ids = set()
        for t, group in progress(df.groupby('t', sort=True)):
            stack = mask[t]
            if isinstance(stack, da.Array):
                stack = stack.compute()

            coords = tuple(
                group[v].values.round().astype(int) for v in columns[2:]
            )
            selected = stack[coords] > 0
            for v in group.loc[selected, 'TrackID']:
                selected_track_ids.add(v)
        
        
        selected_track_ids = self._propagate_selection(selected_track_ids)
        graph = {
            n: p for n, p in self._tracks_layer.value.graph.items()
            if n in selected_track_ids
        }
        # TODO: update parents `p` so they only contain the filtered values
        data = df.loc[df['TrackID'].isin(selected_track_ids)]
        self._viewer.add_tracks(data=data, graph=graph, name='Selection')
    
    def _propagate_selection(self, selected_track_ids: Set[int]) -> Set[int]:
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
                    selected_track_ids.add(track_id)

        return selected_track_ids

    @staticmethod
    def _create_inv_graph(graph: Dict):
        inv_graph = {}
        for node, parents in graph.items():
            for parent in parents:
                if parent in inv_graph:
                    inv_graph[parent].append(node)
                else:
                    inv_graph[parent] = [node]

    def _set_button_status(self, layer: Layer) -> None:
        self._select_button.enabled = self._button_status()

    def _button_status(self) -> bool:
        return (
            self._tracks_layer.value is not None and 
            self._mask_layer.value is not None and
            self._segms_layer.value is not None
        )
