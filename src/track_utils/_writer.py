from napari_plugin_engine import napari_hook_implementation

from typing import Dict, Optional, List
from numpy.typing import ArrayLike

import math as m
import pandas as pd
from ._reader import TRACKS_HEADER


def tracks_to_dataframe(
    data: ArrayLike,
    features: Optional[Dict[str, ArrayLike]] = None,
    graph: Optional[Dict[int, List[int]]] = None,
) -> Optional[pd.DataFrame]:

    if data.ndim != 2 or data.shape[1] > 5 or data.shape[1] < 4:
        return None

    header = list(TRACKS_HEADER)
    if data.shape[1] == 4:
        header.remove('z')
    
    df = pd.DataFrame(data, columns=header)

    if graph:
        nan = (m.nan,)
        df['ParentTrackID'] = df['TrackID'].map(lambda k: graph.get(k, nan)[0])
    
    if features:
        for k, v in features.items():
            df[k] = v

    return df


@napari_hook_implementation
def napari_write_tracks(path: str, data: ArrayLike, meta: Dict) -> Optional[str]:
    df = tracks_to_dataframe(data, meta.get('properties'), meta.get('graph'))
    if df is None:
        return None
    df.to_csv(path, index=False, header=True)
    return path
