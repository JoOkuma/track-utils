
from napari_plugin_engine import napari_hook_implementation
from track_utils.editracks import EditTracks
from track_utils.findtracks import FindTracks
from track_utils.labeltracks import LabelTracks
from track_utils.selecttracks import SelectTracks
from track_utils.splittracks import SplitTracks


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [EditTracks, FindTracks, LabelTracks, SelectTracks, SplitTracks]
