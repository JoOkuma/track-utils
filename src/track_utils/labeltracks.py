
import napari
from napari.layers import Tracks
from napari.layers.tracks._managers import InteractiveTrackManager
from napari.utils.progress import progress

from track_utils.selecttracks import SelectTracks
from magicgui.widgets import LineEdit


class LabelTracks(SelectTracks):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(viewer=viewer)

        self._select_button.text = 'Label'
        self._feature_name = LineEdit(label='Feature Name:', value='labels')

        self.append(self._feature_name)

    def _on_select_clicked(self) -> None:
        track_layer: Tracks = self._tracks_layer.value
        manager: InteractiveTrackManager = track_layer._manager

        df = track_layer.features
        df.set_index('NodeID', drop=False, inplace=True)

        selected_track_ids = self._find_colliding_tracks(df)
        labeled = df['track_id'].isin(selected_track_ids)

        name = self._feature_name.value
        for node_id, label in progress(zip(df['NodeID'], labeled)):
            manager._get_node(node_id).features[name] = int(label)

        manager._is_serialized = False
        track_layer.events.properties()
        track_layer.events.color_by()
