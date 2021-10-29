
from typing import Any
import napari
from napari.layers import Tracks
from magicgui.widgets import Container, create_widget, ComboBox, PushButton
import numpy as np


class SplitTracks(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        self._tracks_layer = create_widget(annotation=Tracks, name='Tracks')
        self._tracks_layer.changed.connect(self._on_tracks_changed)
        self._tracks_layer.changed.connect(self._validate_split_button)
        self.append(self._tracks_layer)

        self._category = ComboBox(name='Category')
        self.append(self._category)

        self._split_button = PushButton(text='Split', enabled=False)
        self._split_button.changed.connect(self._on_split)
        self.append(self._split_button)
    
    def _on_tracks_changed(self, tracks: Tracks) -> None:
        if tracks is None:
            return
        tracks.events.properties.connect(self._update_categories)
        if 'track_id' not in tracks.properties:
            tracks.properties.update({'track_id': tracks.data[:, 0]})
            tracks.events.properties()
        else:
            self._update_categories()   # force update

    def _update_categories(self, _: Any = None) -> None:
        self._category.choices = tuple(self._tracks_layer.value.properties.keys())
        print('Updated categories', self._category.choices)
    
    def _validate_split_button(self, tracks: Tracks) -> None:
        self._split_button.enabled = self._tracks_layer.value is not None
    
    def _on_split(self) -> None:
        tracks = self._tracks_layer.value
        props = tracks.properties
        if self._category.value not in props:
            return
        
        colname = self._category.value
        column = props[colname]

        for key, indices in zip(np.unique(column, return_inverse=True)):
            split_props = {k: v[indices] for k, v in props}
            self._viewer.add_tracks(
                data=tracks.data[indices, :],
                name=f'{colname}_{key}',
                properties=split_props,
            )
