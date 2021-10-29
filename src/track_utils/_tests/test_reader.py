from pathlib import Path
from track_utils import napari_get_reader
import numpy as np
import pandas as pd


def _build_tracks(n_nodes: int = 10) -> pd.DataFrame:
    coordinates = np.random.rand(n_nodes, 3)
    tracks_data = np.zeros((n_nodes, 5))
    tracks_data[:, 2:] = coordinates

    tracks_data[:n_nodes // 2, 0] = 1
    tracks_data[n_nodes // 2:, 0] = 2
    tracks_data[:n_nodes // 2, 1] = np.arange(n_nodes // 2)
    tracks_data[n_nodes // 2:, 1] = np.arange(n_nodes - n_nodes // 2)

    return pd.DataFrame(tracks_data, columns=['TrackID', 't', 'z', 'y', 'x'])


def test_reader(tmp_path):
    reader = napari_get_reader('tracks.csv')
    assert reader is None

    path = tmp_path / 'good_tracks.csv'
    tracks = _build_tracks()
    tracks['NodeID'] = np.arange(len(tracks)) + 1
    tracks['Labels'] = np.random.randint(2, size=len(tracks))
    tracks.to_csv(path)

    reader = napari_get_reader(path)
    assert callable(reader)

    data, kwargs, type = reader(path)[0]
    assert type == 'tracks'

    props = kwargs['properties']

    assert np.allclose(props['NodeID'], tracks['NodeID'])
    assert np.allclose(props['Labels'], tracks['Labels'])
    assert np.allclose(data, tracks[['TrackID', 't', 'z', 'y', 'x']])


def test_non_existing_track():
    reader = napari_get_reader('tracks.csv')
    assert reader is None


def test_wrong_columns_track(tmp_path: Path):
    reader = napari_get_reader('tracks.csv')
    assert reader is None

    tracks = _build_tracks()
    path = tmp_path / 'bad_tracks.csv'
    tracks.rename(columns={'TrackID': 'id'}, inplace=True)
    tracks.to_csv(path)
    reader = napari_get_reader(path)
    assert reader is None
