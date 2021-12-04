from pathlib import Path
from track_utils import napari_get_reader
import numpy as np

from .utils import build_tracks


def test_reader(tmp_path):
    reader = napari_get_reader('tracks.csv')
    assert reader is None

    path = tmp_path / 'good_tracks.csv'
    tracks = build_tracks()
    tracks['NodeID'] = np.arange(len(tracks)) + 1
    tracks['Labels'] = np.random.randint(2, size=len(tracks))
    tracks.to_csv(path, index=False)

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

    tracks = build_tracks()
    path = tmp_path / 'bad_tracks.csv'
    tracks.rename(columns={'TrackID': 'id'}, inplace=True)
    tracks.to_csv(path, index=False)
    reader = napari_get_reader(path)
    assert reader is None
