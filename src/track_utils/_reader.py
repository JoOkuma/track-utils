import os
import pandas as pd
import numpy as np
import math as m

from pathlib import Path
from napari_plugin_engine import napari_hook_implementation


TRACKS_HEADER = ('TrackID', 't', 'z', 'y', 'x')


@napari_hook_implementation
def napari_get_reader(path):
    if isinstance(path, list):
        path = path[0]
    
    if isinstance(path, str):
        path = Path(path)
    
    if not path.name.endswith('.csv') or not os.path.exists(path):
        return None

    header = pd.read_csv(path, nrows=0).columns.tolist()
    for colname in TRACKS_HEADER:
        if colname != 'z' and colname not in header:
            return None

    return reader_function


def read_csv(path: str):
    df = pd.read_csv(path)

    data = []
    for colname in TRACKS_HEADER:
        try:
            data.append(df[colname])
        except KeyError:
            if colname != 'z':
                raise KeyError(f'{colname} not found in .csv header.')

    data = np.stack(data).T

    props = {
        colname: df[colname]
        for colname in df.columns
        if colname not in TRACKS_HEADER
    }
    
    kwargs = {'features': pd.DataFrame(props)}

    # optional step:
    if 'ParentTrackID' in df.columns:
        graph_df = df.drop_duplicates(('TrackID', 'ParentTrackID'))
        kwargs['graph'] = {
            track: [parent] for track, parent in zip(graph_df['TrackID'], graph_df['ParentTrackID'])
            if not m.isnan(parent)
        }

    return (data, kwargs, 'tracks')


def reader_function(path):
    paths = [path] if isinstance(path, (str, Path)) else path
    return [read_csv(p) for p in paths]
