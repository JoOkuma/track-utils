
from typing import Dict, List
import re
import numpy as np
from scipy.sparse import csr_matrix

from zarr import Array
from zarr.storage import listdir
from typing import Iterator, Tuple


def get_initialized_keys(array: Array) -> Iterator[str]:
    # key pattern for chunk keys
    prog = re.compile(r'\.'.join([r'\d+'] * min(1, array.ndim)))

    # yield chunk keys
    for k in listdir(array.chunk_store, array._path):
        if prog.match(k):
            yield k


def zarr_key_to_slice(array: Array, key: str) -> Tuple[slice]:
    start = np.array(key.split('.')).astype(int)
    start *= array.chunks
    stop = np.minimum(start + array.chunks, array.shape)
    return tuple(slice(s, e) for s, e in zip(start, stop))


def graph_to_csr(graph: Dict[int, List[int]]) -> csr_matrix:
    
    rows = []
    cols = []
    for src, dsts in graph.items():
        rows += [src] * len(dsts)
        cols += dsts
    
    data = np.ones(len(rows), dtype=bool)
    return csr_matrix((data, (rows, cols)))

