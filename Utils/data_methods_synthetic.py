import numpy as np
import torch
from sklearn.model_selection import train_test_split


def generate_anomalies(x, num: int, delta: float = 0.05, max_dim=None, min_dim=None, return_anom_only=False, seed=23):
    """
    MADI Paper from Google [ICML]
    https://github.com/google/madi/blob/master/src/madi/utils/sample_utils.py

    Creates anomalous samples from the cuboid bounded by +/- delta.

  Where, [min - delta, max + delta] for each of the dimensions.
  The positive sample, pos_sample is a pandas DF that has a column
  labeled 'class_label' where 1.0 indicates Normal, and
  0.0 indicates anomalous.

  Args:
    x: normalised DF / array-like with numeric dimensions
    num: number points to be returned
    delta: fraction of [max - min] to extend the sampling.
    max_dim: array (d,) shape, pre-specified maximum across dimensions
    min_dim: array (d,) shape, pre-specified minimum across dimensions

  Returns:
    An (x, y) tuple, where x is stacked normal and generated anomalies, and y is 1 for normal, 0 for anomaly
    """

    if max_dim is None:
        max_dim = np.max(x, axis=0)
    if min_dim is None:
        min_dim = np.min(x, axis=0)

    interval = max_dim - min_dim
    buffer = delta * interval

    np.random.seed(seed)
    anomalies_generated = np.random.uniform(low=min_dim - buffer, high=max_dim + buffer, size=(num, len(max_dim)))

    if return_anom_only:
        return anomalies_generated

    return np.vstack((x, anomalies_generated)), np.hstack((np.ones(len(x)), np.zeros(num)))


def get_dataloader(x, y, batch_size=128):
    tensor_x = torch.Tensor(x)  # transform to torch tensor
    tensor_y = torch.Tensor(y)

    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader
