import numpy as np
import pandas as pd

from track_utils.selecttracks import SelectTracks
import track_utils

MY_PLUGIN_NAME = "track-utils"


def _create_data():
    segms = np.zeros((5, 10, 10), dtype=int)

    n_tracks = 5

    for i in range(n_tracks):
        j = 2 * i
        segms[:, j:j+2, j:j+2] = i + 1
    
    # track split from 2 -> (3, 4)
    j = 2 * 4
    segms[:3, j:j+2, j:j+2] = 0

    j = 2 * 3
    segms[:3, j:j+2, j:j+2] = 0

    j = 2 * 2
    segms[3:, j:j+2, j:j+2] = 0

    dfs = []
    max_node_id = 1
    instance_segms = np.zeros_like(segms)
    for i in range(n_tracks):
        t, y, x = np.where(segms == i + 1)
        df = pd.DataFrame({'t': t, 'y': y + 0.5, 'x': x + 0.5})  # centering to pixels
        df = df.groupby(t).aggregate(np.mean)
        df['TrackID'] = i + 1
        df['NodeID'] = np.arange(len(df)) + max_node_id
        max_node_id = df['NodeID'].max() + 1
        instance_segms[segms == i + 1] = np.repeat(df['NodeID'].values, 4)
        dfs.append(df)
    
    df = pd.concat(dfs)
    df.reset_index()

    mask = segms == 4
    return mask, instance_segms, df


def test_edit_tracks_load(make_napari_viewer, napari_plugin_manager):
    napari_plugin_manager.register(track_utils, name=MY_PLUGIN_NAME)

    viewer = make_napari_viewer()
    widget: SelectTracks = viewer.window.add_plugin_dock_widget(
        plugin_name='track-utils', widget_name="Select Tracks"
    )[1]

    mask, instance_segms, df = _create_data()
    print(df)

    widget._segms_layer.value = viewer.add_labels(instance_segms)
    widget._mask_layer.value = viewer.add_labels(mask)
    tracks_layer = viewer.add_tracks(
        df[['TrackID', 't', 'y', 'x']].values,
        features={'NodeID': df['NodeID'].values},
        graph = {5: [3], 4: [3]},
    )
    tracks_layer.editable = True

    widget._on_select_clicked()

    new_layer = viewer.layers[-1]
    assert np.all(np.unique(new_layer.data[:, 0]) == [12, 14, 16])
    assert new_layer.graph == {14: [12], 16: [12]}
