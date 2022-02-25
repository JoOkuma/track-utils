
from napari_plugin_engine import napari_hook_implementation
from .editracks import EditTracks
from .findtracks import FindTracks
from .selecttracks import SelectTracks
from .splittracks import SplitTracks


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [EditTracks, FindTracks, SelectTracks, SplitTracks]
