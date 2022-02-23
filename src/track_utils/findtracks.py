
from typing import Callable, Optional

import numpy as np

import napari
from napari.layers.tracks import Tracks
from napari.layers.tracks._managers._interactive_track_manager import Node

from qtpy.QtWidgets import QWidget, QVBoxLayout
from psygnal import Signal
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, create_widget

from .editracks import EditTracks
from .features import difference
from .property_plotter import PropertyPlotter


def _neigh_relative_speed():
    pass


NO_TRACK = -1


class FindTracks(QWidget):
    _index_changed = Signal()

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self._track_layer_combobox = create_widget(annotation=Tracks, name="Track Layer:")
        self._track_layer_combobox.reset_choices()
        self._viewer.layers.events.inserted.connect(self._track_layer_combobox.reset_choices)
        self._viewer.layers.events.removed.connect(self._track_layer_combobox.reset_choices)

        self._nodes_id_ordering = []
        self._ordering_index = 0

        self._layout.addWidget(self._track_layer_combobox.native)
        self._layout.addWidget(self.search_priority.native)

        container = Container(layout="horizontal")
        self._prev_button = PushButton(text="Prev", enabled=False)
        self._prev_button.changed.connect(lambda _: self._increment_ordering_index(-1))
        container.append(self._prev_button)

        self._next_button = PushButton(text="Next", enabled=False)
        self._next_button.changed.connect(lambda _: self._increment_ordering_index(+1))
        container.append(self._next_button)

        self._validate_buttons()

        self._layout.addWidget(container.native)

        self._props_plotter = PropertyPlotter(self._viewer)
        self._props_plotter.data_selector.new_selection.connect(self.on_plot_selection_changed)
        self._layout.addWidget(self._props_plotter)
        self._displayed_track_id = NO_TRACK

        self._index_changed.connect(self._validate_buttons)
        self._index_changed.connect(self._select_node)

        track_editor = self._edit_tracks_widget()
        if track_editor is not None:
            pass
        
        track_editor.events.selected_node.connect(self.on_canvas_selection_changed)

    def _edit_tracks_widget(self) -> Optional[EditTracks]:
        dock_widget = self._viewer.window._dock_widgets.get('track-utils: Edit Tracks')
        if dock_widget is not None:
            return dock_widget.widget()._magic_widget
        return None
    
    def _get_node(self, index: int) -> Node:
        manager = self._track_layer_combobox.value._manager
        return manager._get_node(index)
    
    def on_canvas_selection_changed(self, event) -> None:
        if event.value is None:
            return

        node = event.value
        if self._displayed_track_id != node.features.get('track_id'):
            # if df haven't been set yet
            self.plot_node_subtree(node.index)

        selection = self._props_plotter.data_selector.sele
        if selection is None:
            return

        self._props_plotter.picker.x_picker.setCurrentText('time')
        selection.setValue(float(event.value.time))

    def on_plot_selection_changed(self, value=None) -> None:
        if value is None:
            return
        track_editor = self._edit_tracks_widget()
        if track_editor is None:
            return
        x_prop = self._props_plotter.picker.x
        argmin = np.argmin(np.abs(x_prop - value))
        
        index = self._props_plotter.picker.df['index'].iloc[argmin]
        track_editor.selected_node = self._get_node(index)

    def _select_node(self) -> None:
        selected_index = self._nodes_id_ordering[self._ordering_index]
        track_editor = self._edit_tracks_widget()
        if track_editor is None:
            return

        self.plot_node_subtree(selected_index)

        try:
            track_editor.selected_node = self._get_node(selected_index)
        except IndexError:
            pass
    
    def plot_node_subtree(self, index: int) -> None:
        if self._track_layer_combobox.value is None:
            return

        manager = self._track_layer_combobox.value._manager
        df = manager._connected_component(index, return_df=True)

        df.set_index('index', inplace=True)  # hack necessary, picker reverts this
        df['moviment'] = difference(df, ['z', 'y', 'x'])
        df = df.astype(float)
        self._props_plotter.picker.set_dataframe(df)
        self._props_plotter.picker.x_picker.setCurrentText('time')
        self._displayed_track_id = df['track_id'].iloc[0]
   
    @magicgui(
        method = {
            "choices": [
            ("Neigh. Relative Speed", _neigh_relative_speed),
        ]},
        call_button="Search",
    )
    def search_priority(self, method: Callable) -> None:
        print("searching with", method.__name__)
        self._nodes_id_ordering = [50, 100, 150]

        self._ordering_index = 0
        self._index_changed.emit()
    
    def _increment_ordering_index(self, step: int):
        self._ordering_index += step
        self._index_changed.emit()
    
    def _validate_buttons(self) -> None:
        self._next_button.enabled = self._ordering_index + 1 < len(self._nodes_id_ordering)
        self._prev_button.enabled = self._ordering_index - 1 >= 0
