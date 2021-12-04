import numpy as np
import pandas as pd


def build_tracks(n_nodes: int = 10) -> pd.DataFrame:
    coordinates = np.random.rand(n_nodes, 3)
    tracks_data = np.zeros((n_nodes, 5))
    tracks_data[:, 2:] = coordinates

    tracks_data[:n_nodes // 2, 0] = 1
    tracks_data[n_nodes // 2:, 0] = 2
    tracks_data[:n_nodes // 2, 1] = np.arange(n_nodes // 2)
    tracks_data[n_nodes // 2:, 1] = np.arange(n_nodes - n_nodes // 2)

    return pd.DataFrame(tracks_data, columns=['TrackID', 't', 'z', 'y', 'x'])
