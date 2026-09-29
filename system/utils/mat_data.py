from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


@dataclass
class MultiViewData:
    name: str
    views: list
    labels: np.ndarray

    @property
    def num_samples(self):
        return len(self.labels)

    @property
    def view_dims(self):
        return [view.shape[1] for view in self.views]

    @property
    def num_clusters(self):
        return len(np.unique(self.labels))


class MultiViewSubset(Dataset):
    def __init__(self, data, indices):
        self.views = [torch.from_numpy(view[indices]) for view in data.views]
        self.labels = torch.from_numpy(data.labels[indices]).long()
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return tuple(view[index] for view in self.views), self.labels[index], self.indices[index]


def load_multiview_mat(path, name=None, normalization="standard"):
    path = Path(path)
    payload = loadmat(path)
    view_key = "X" if "X" in payload else "data" if "data" in payload else None
    label_key = next(
        (key for key in ("Y", "labels", "gt") if key in payload),
        None,
    )
    if view_key is None or label_key is None:
        raise KeyError(f"{path} must contain X/Y, data/labels, or X/gt")

    raw_views = np.asarray(payload[view_key], dtype=object).reshape(-1)
    labels = np.asarray(payload[label_key]).reshape(-1)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)

    views = []
    for raw_view in raw_views:
        view = np.asarray(raw_view)
        if view.ndim != 2:
            raise ValueError(f"Every view must be 2-D, got {view.shape} in {path}")
        if view.shape[0] != len(labels) and view.shape[1] == len(labels):
            view = view.T
        if view.shape[0] != len(labels):
            raise ValueError(f"View sample count {view.shape[0]} != label count {len(labels)}")
        view = view.astype(np.float32, copy=False)
        if not np.isfinite(view).all():
            raise ValueError(f"View in {path} contains NaN or infinity")
        views.append(_normalize_view(view, normalization))

    return MultiViewData(name=name or path.stem, views=views, labels=labels)


def _normalize_view(view, method):
    if method == "standard":
        mean = view.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        std = view.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        return np.ascontiguousarray((view - mean) / std, dtype=np.float32)
    if method == "minmax":
        minimum = view.min(axis=0, keepdims=True)
        scale = view.max(axis=0, keepdims=True) - minimum
        scale[scale < 1e-6] = 1.0
        return np.ascontiguousarray((view - minimum) / scale, dtype=np.float32)
    if method == "l2":
        norm = np.linalg.norm(view, axis=1, keepdims=True)
        norm[norm < 1e-6] = 1.0
        return np.ascontiguousarray(view / norm, dtype=np.float32)
    if method in (None, "none"):
        return np.ascontiguousarray(view, dtype=np.float32)
    raise ValueError(f"Unknown normalization method: {method}")


def balanced_client_indices(num_samples, num_clients, seed):
    """Create balanced random partitions without consulting evaluation labels."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(int(num_samples)).astype(np.int64, copy=False)
    return [np.ascontiguousarray(part) for part in np.array_split(indices, num_clients)]
