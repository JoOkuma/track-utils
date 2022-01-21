from typing import List

import numpy as np
import pandas as pd


def difference(tracks: pd.DataFrame, columns: List[str] = None) -> np.ndarray:
    diff = np.zeros(tracks.shape[0])
    tracks = tracks.reset_index(inplace=False)

    for _, group in tracks.groupby('track_id'):
        if columns is not None:
            group = group[columns]
        diff[group.index[1:]] = np.linalg.norm(group.values[1:] - group.values[:-1], axis=1)
    
    return diff
