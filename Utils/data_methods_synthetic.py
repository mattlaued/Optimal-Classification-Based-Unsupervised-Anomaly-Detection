import numpy as np
import torch
from sklearn.model_selection import train_test_split


def generate_synthetic_data(seed, n_samples, n_dimensions, m_nonzero_entries, add=False):
    dim1, dim2 = n_dimensions
    np.random.seed(seed)

    # Generate random vectors for each sample
    random_vectors = np.random.rand(n_samples, dim2)

    # Create matrix A
    #     A = np.ones((dim1, dim2)) * 1 / m_nonzero_entries
    A = np.zeros((dim1, dim2))

    # Generate random indices for non-zero entries in the matrix A
    #     non_zero_indices = np.random.choice(dim1, size=(m_nonzero_entries), replace=False)
    non_zero_indices = np.random.rand(*A.shape).argsort(1)[:, :m_nonzero_entries]
    for i, row in enumerate(non_zero_indices):
        A[i, row] = 1 / m_nonzero_entries

    # Multiply random vectors by matrix A
    synthetic_data = np.dot(A, random_vectors.T).T

    if add:
        addition = np.zeros((n_samples, dim1))
        add_col = np.random.randint(2, size=n_samples)
        addition[:, -1] = add_col
        synthetic_data += addition

    return synthetic_data


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


def rejection_sampling_uniform(num_samples, dim, target_density, seed1=0, seed2=10, to_cuda=True):
    """
    Rejection sampling with target density g and original density f as uniform on hypercube
    Args:
        num_samples:
        target_density: g
        seed1: seed for sample generation
        seed2: seed for uniform generation for rejection statistic
        to_cuda: move

    Returns:
    Samples from g
    """

    np.random.seed(seed1)
    samples = np.random.uniform(size=(num_samples, dim))
    # density of points drawn from uniform is constant. can change this for different original density f
    density = 1.
    # scale in closed form for uniform
    scale = 1.

    if to_cuda:
        samples = torch.from_numpy(samples).float().to("cuda")

    sample_density = target_density(samples)

    # tau = g(x) / (scale x f(x))
    tau = sample_density / (scale * density)
    if to_cuda:
        tau = tau.cpu().detach().numpy()
    np.random.seed(seed2)
    u = np.random.uniform(size=(num_samples, 1))
    idx = (u > tau).squeeze()

    return samples[idx]


def labelling_fn(x, target_density, s, rho=None, threshold=0.5):
    """
    Labelling function
    Args:
        x:
        target_density: density function of x, h(x)
        s: scaling parameter. can be defined explicitly or as function of rho for s = 1/(1+rho)
        rho: threshold density to be considered normal
        threshold: threshold for soft label

    Returns: 1 for normal, 0 for anomaly

    """

    if rho is not None:
        s = 1 / (1 + rho)

    density = target_density(x)
    soft_label = s * density / (s * density + 1 - s)
    return soft_label >= threshold


def pseudo_nn_density_function(x):
    x_ = np.maximum(0., 8 - 64 * np.abs(x - 0.5))
    return np.prod(x_, axis=1)


def get_data_and_label(num_training_normal, num_train_normal, num_val_normal, num_test_normal,
                       num_training_anom, num_train_anom, num_val_anom, num_test_anom,
                       n_dimensions, m_nonzero_entries=2, add=False, target_density=False, rejection_sampling=False,
                       seed=123, seed_test=623, seed_shuffle=1234, **kwargs):
    if target_density and not rejection_sampling:
        # we will use num_training_normal and num_test_normal and ignore the anom info
        s = kwargs.get('s', 0.5)
        rho = kwargs.get('rho', None)
        threshold = kwargs.get('threshold', 0.5)
        np.random.seed(seed)
        training_data = np.random.uniform(low=0., high=1., size=(num_training_normal, n_dimensions))
        training_labels = labelling_fn(training_data, target_density=target_density, s=s, rho=rho, threshold=threshold)
        # get validation data with train:val ratio
        x_train, x_val, y_train, y_val = train_test_split(
            training_data, training_labels, test_size=num_val_normal/(num_train_normal+num_val_normal),
            random_state=0)

        np.random.seed(seed_test)
        x_test = np.random.uniform(low=0., high=1., size=(num_test_normal, n_dimensions))
        y_test = labelling_fn(x_test, target_density=target_density, s=s, rho=rho, threshold=threshold)

        print(f"Num Normal (train, val, test): {sum(y_train == 1), sum(y_val == 1), sum(y_test == 1)}")
        print(f"Num Anom   (train, val, test): {sum(y_train == 0), sum(y_val == 0), sum(y_test == 0)}")

    else:
        if target_density:
            x_training_normal = rejection_sampling_uniform(
                num_training_normal, n_dimensions, target_density, seed1=seed, seed2=seed+100, to_cuda=True
            ).cpu().detach().numpy()
            x_test_normal = rejection_sampling_uniform(
                num_training_normal, n_dimensions, target_density, seed1=seed_test, seed2=seed_test+100, to_cuda=True
            ).cpu().detach().numpy()
            num_training_normal = len(x_training_normal)
            num_val_normal = num_training_normal - num_train_normal
            num_test_normal = len(x_test_normal)
            num_test_anom = num_test_normal
        else:
            x_training_normal = generate_synthetic_data(seed, num_training_normal, n_dimensions, m_nonzero_entries, add=add)
            x_test_normal = generate_synthetic_data(seed_test, num_test_normal, n_dimensions, m_nonzero_entries, add=add)
        x_train_normal = x_training_normal[:num_train_normal]
        x_val_normal = x_training_normal[num_train_normal:]

        print(x_train_normal.shape)
        print(x_val_normal.shape)
        print(x_test_normal.shape)

        # Get anoms

        ratio_train = num_training_anom / num_train_normal
        ratio_test = num_test_anom / num_test_normal

        # num_normal = np.sum(y)
        num_generated_anomalies_train = int(ratio_train * num_train_normal)
        num_generated_anomalies_test = int(ratio_test * num_test_normal)

        x_training, y_training = generate_anomalies(
            x_train_normal, num_generated_anomalies_train, delta=0., max_dim=np.ones(n_dimensions),
            min_dim=np.zeros(n_dimensions))
        x_train = x_training[:-num_val_anom]
        y_train = y_training[:-num_val_anom]
        x_val = np.vstack((x_val_normal, x_training[-num_val_anom:]))
        y_val = np.hstack((np.ones(num_val_normal), y_training[-num_val_anom:]))
        x_test, y_test = generate_anomalies(
            x_test_normal, num_generated_anomalies_test, delta=0., max_dim=np.ones(n_dimensions),
            min_dim=np.zeros(n_dimensions))

        np.random.seed(seed_shuffle)
        np.random.shuffle(x_train)
        np.random.seed(seed_shuffle)
        np.random.shuffle(y_train)

    print("x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape")
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test
