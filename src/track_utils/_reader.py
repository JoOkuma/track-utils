import pandas as pd
import numpy as np
from napari_plugin_engine import napari_hook_implementation


@napari_hook_implementation
def napari_get_reader(path):
    if isinstance(path, list):
        path = path[0]

    if not path.endswith(".csv"):
        return None

    return reader_function


def read_csv(path: str):
    df = pd.read_csv(path)

    tracks_header = ('TrackID', 't', 'z', 'y', 'x')

    data = []
    for colname in tracks_header:
        try:
            data.append(df[colname])
        except KeyError:
            if colname != 'z':
                raise KeyError(f'{colname} not found in .csv header.')

    data = np.stack(data).T

    props = {
        colname: df[colname]
        for colname in df.columns
        if colname not in tracks_header
    }
    
    kwargs = {'properties': props}
    return (data, kwargs, 'tracks')


def reader_function(path):
    paths = [path] if isinstance(path, str) else path
    return [read_csv(p) for p in paths]
