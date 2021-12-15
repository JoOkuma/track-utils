import numpy as np

from .utils import build_tracks
import track_utils

MY_PLUGIN_NAME = "track-utils"


def test_edit_tracks_load(make_napari_viewer, napari_plugin_manager):
    napari_plugin_manager.register(track_utils, name=MY_PLUGIN_NAME)

    viewer = make_napari_viewer()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name='track-utils', widget_name="Edit Tracks"
    )

    data = build_tracks()
    viewer.add_tracks(data.values)
    widget._on_load()
    widget._select(5)
