import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from Utils.data_methods_synthetic import generate_anomalies, get_dataloader
from torch.utils.data import TensorDataset
import os


def get_df(path, columns, drop, read="csv", header=None):
    """
    Function to get a dataframe from a csv file.
    Args:
        path: file name of csv file
        columns: column names of the csv file, or None if alr provided
        drop: False or list of columns to drop
        read: filetype (csv, parquet, etc)

    Returns:

    """
    if read == "csv":
        reader = pd.read_csv
    else:
        reader = pd.read_parquet
    if type(path) is str:
        df = reader(path, header=header)
    else:
        dfs = [reader(pth, header=header) for pth in path]
        # concat vertically i.e. along columns
        df = pd.concat(dfs, axis=1)
    if columns is not None:
        df.columns = columns

    if drop:
        df = df.drop(columns=drop)

    return df.drop_duplicates().reset_index(drop=True)


def feat_eng(df, test_df, features_to_encode, numeric_features):
    """
    Feature engineering function. One-hot encodes categorical columns.
    Args:
        df:
        test_df:
        features_to_encode:
        numeric_features:

    Returns:

    """
    #     https://www.kaggle.com/code/avk256/nsl-kdd-anomaly-detection/notebook

    if len(features_to_encode) > 0:

        # not all of the features are in the test set, so we need to account for diffs
        column_diffs, encoded, test_encoded_base = get_train_cols_not_in_test(df, test_df, features_to_encode)
        test_index = np.arange(len(test_df.index))
        diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)

        # we'll also need to reorder the columns to match, so let's get those
        column_order = encoded.columns.to_list()

        # append the new columns
        test_encoded_temp = test_encoded_base.join(diff_df)

        # reorder the columns
        test_final = test_encoded_temp[column_order].fillna(0)

        # model to fit/test
        to_fit = encoded.join(df[numeric_features])
        test_set = test_final.join(test_df[numeric_features])

    else:
        to_fit = df[numeric_features]
        test_set = test_df[numeric_features]

    return to_fit, test_set


def get_train_cols_not_in_test(df, test_df, features_to_encode):
    # get the intial set of encoded features and encode them
    # if len(features_to_encode) > 0:
    encoded = pd.get_dummies(df[features_to_encode])
    test_encoded_base = pd.get_dummies(test_df[features_to_encode])

    # not all of the features are in the test set, so we need to account for diffs
    column_diffs = sorted(list(set(encoded.columns.values) - set(test_encoded_base.columns.values)))
    # confirm that all test cols are in training
    assert len(set(test_encoded_base.columns.values) - set(encoded.columns.values)) == 0

    # else:
    #     encoded = pd.DataFrame()
    #     test_encoded_base = pd.DataFrame()
    #     column_diffs = []

    return column_diffs, encoded, test_encoded_base


def get_numeric(df, test_df, features_to_encode, numeric_features, map_attack, raw_label_col="attack",
                normal_label="normal"):
    """
    get numeric training and testing dataframe, without labels
    Args:
        df: training dataframe
        test_df: testing dataframe
        features_to_encode: categorical features for one-hot encoding
        numeric_features: numerical features to use
        map_attack: function to map attack to numeric attack type
        raw_label_col: column name in df that has attack names
        normal_label: label of normal data in the label column

    Returns: preprocessed (train_df, test_df)

    """
    # map normal to 0, all attacks to 1
    is_attack = df[raw_label_col].map(lambda a: 0 if a == normal_label else 1)
    test_attack = test_df[raw_label_col].map(lambda a: 0 if a == normal_label else 1)

    # data_with_attack = df.join(is_attack, rsuffix='_flag')
    df['attack_flag'] = is_attack
    test_df['attack_flag'] = test_attack

    # map normal to 1, all attacks to 0
    is_normal = df[raw_label_col].map(lambda a: 1 if a == normal_label else 0)
    test_normal = test_df[raw_label_col].map(lambda a: 1 if a == normal_label else 0)

    df['normal_flag'] = is_normal
    test_df['normal_flag'] = test_normal

    # map the data and join to the data set
    attack_map = df[raw_label_col].apply(map_attack)
    df['attack_map'] = attack_map

    test_attack_map = test_df[raw_label_col].apply(map_attack)
    test_df['attack_map'] = test_attack_map

    return feat_eng(df, test_df, features_to_encode, numeric_features)


def get_x_y(df, data, classes, index_match_col='attack_map', label_col='normal_flag'):
    """
    Get data and labels of classes
    Args:
        df: original dataframe
        data: features used
        classes: 'all' or list of classes to use
        index_match_col: if data record has value in classes along this column, we use the data record
        label_col: column to collect labels from

    Returns:

    """
    if type(classes) is list:
        indices = df[index_match_col].isin(classes)
        x = data[indices]
        y = df[label_col][indices]
    elif classes == "all":
        x = data
        y = df[label_col]
    else:
        raise Exception("list or 'all'")

    return x.to_numpy(), y.to_numpy()


def preprocess(df, test_df, features_to_encode, numeric_features, training_classes, map_attack, raw_label_col,
               normal_label="normal", scaler=MinMaxScaler(feature_range=(0, 1)), test_classes="all",
               train_index_match_col='attack_map', train_label_col='normal_flag',
               test_index_match_col='attack_map', test_label_col='attack_map'):

    # get one-hot cols
    one_hot_col_len = []
    for feature in features_to_encode:
        one_hot_col_len.append(len(df[feature].unique()))

    data_train, data_test = get_numeric(
        df, test_df, features_to_encode, numeric_features, map_attack=map_attack, raw_label_col=raw_label_col,
        normal_label=normal_label)

    print("Training Features")
    print(data_train.head())
    print("Testing Features")
    print(data_test.head())

    x_training, y_training = get_x_y(
        df, data_train, classes=training_classes, index_match_col=train_index_match_col, label_col=train_label_col)
    X = scaler.fit_transform(x_training)

    np.random.seed(0)
    np.random.shuffle(X)
    np.random.seed(0)
    np.random.shuffle(y_training)

    # for testing, get labels which are int: tell us what type of attack
    x_testing, y_test = get_x_y(test_df, data_test, classes=test_classes,
                                index_match_col=test_index_match_col, label_col=test_label_col)
    x_test = scaler.transform(x_testing)

    # # validation split
    # if val_split > 0:
    #     num_train = int(len(X) * (1 - val_split))
    #     x_train_real = X[:num_train]
    #     y_train_real = y_training[:num_train]
    #     x_val_real = X[num_train:]
    #     y_val_real = y_training[num_train:]
    # else:
    #     # num_train = None
    #     x_train_real = X
    #     y_train_real = y_training
    #     x_val_real = None
    #     y_val_real = None

    return X, y_training, x_test, y_test, one_hot_col_len


def add_synthetic_anom_to_test_data(x_test, y_test, synthetic_anom_ratio, one_hot_col_len, seed_anom_generation=236):
    """

    Args:
        x_test: data
        y_test: 0 for normal, 1/2/3/.../k for different attacks
        synthetic_anom_ratio: proportion of synthetic anomalies to generate wrt num normal data
        one_hot_col_len: categorical columns
        seed_anom_generation: seed for generating synthetic anomaly

    Returns: tuple of x_test concat w synthetic anoms, y_test concat w synthetic anom label of k+1

    """
    normal_indices = (y_test == 0)
    num_normal = np.sum(normal_indices)
    x_test_normal = x_test[normal_indices]
    y_test_normal = np.ones(num_normal)

    # normal x labeled 1, anom labeled 0
    x, y, _, _ = combine_real_synthetic(
        x_test_normal, y_test_normal, None, None, synthetic_anom_ratio=synthetic_anom_ratio,
        one_hot_col_len=one_hot_col_len, seed_anom_generation=seed_anom_generation)

    # synthetic anoms have different attack label
    synthetic_anom_label = (1 + np.max(y_test))
    synthetic_anoms = x[num_normal:]
    synthetic_anom_labels = np.ones(len(synthetic_anoms)) * synthetic_anom_label

    return np.vstack((x_test, synthetic_anoms)), np.hstack((y_test, synthetic_anom_labels))


def validation_split(X, y, val_split=0.2, seed=1234):

    np.random.seed(seed)
    x_training = np.random.permutation(X)
    np.random.seed(seed)
    y_training = np.random.permutation(y)

    # validation split
    if val_split > 0:
        num_train = int(len(x_training) * (1 - val_split))
        x_train_real = x_training[:num_train]
        y_train_real = y_training[:num_train]
        x_val_real = x_training[num_train:]
        y_val_real = y_training[num_train:]
    else:
        # num_train = None
        x_train_real = x_training
        y_train_real = y
        x_val_real = None
        y_val_real = None

    return x_train_real, y_train_real, x_val_real, y_val_real


def combine_real_synthetic(x_train_real, y_train_real, x_val_real, y_val_real, synthetic_anom_ratio,
                           synthetic_val_anom_constant=False, one_hot_col_len=[], binary_cols=False, delta=0.,
                           seed_anom_generation=23):
    """
    Adds synthetic anomalies to training and validation data, where 0 is anom and 1 is normal
    Args:
        x_train_real:
        y_train_real:
        x_val_real:
        y_val_real:
        synthetic_anom_ratio:
        synthetic_val_anom_constant:
        one_hot_col_len:
        delta:
        seed_anom_generation:

    Returns: x_train, y_train, x_val, y_val

    """

    n_dimensions = x_train_real.shape[-1]
    num_ohe_values = sum(one_hot_col_len)
    if binary_cols:
        if type(binary_cols) is bool:
            # Bernoulli sampling for binary variables --- check for number of unique values at 0 and 1 (ignore OHE vars)
            binary_cols = (~((~np.isin(x_train_real[:, num_ohe_values:], [0, 1])).any(axis=0))).nonzero()[0]
    num_numeric = n_dimensions - num_ohe_values #- len(binary_cols)
    num_training = len(x_train_real)
    if x_val_real is not None:
        num_val = len(x_val_real)
    else:
        num_val = 0
    # num_data = len(x_train_real) + len(x_val_real)

    # generate synthetic anoms
    if synthetic_anom_ratio > 0:
        num_generated_anomalies_training = int(synthetic_anom_ratio * num_training)
        if synthetic_val_anom_constant:
            num_generated_anomalies_val = num_val
        else:
            num_generated_anomalies_val = int(synthetic_anom_ratio * num_val)
        num_generated_anomalies_train = num_generated_anomalies_training + num_generated_anomalies_val
        anoms_synthetic_training_numeric = generate_anomalies(
            None, num_generated_anomalies_train, delta=delta, max_dim=np.ones(num_numeric),
            min_dim=np.zeros(num_numeric), return_anom_only=True, seed=seed_anom_generation)

        if binary_cols is not False and binary_cols is not None:
            # Bernoulli sampling for binary variables --- can check for number of unique values at 0 and 1
            # sample bernoulli variables
            bernoulli_vars = np.random.binomial(n=1, p=0.5, size=(num_generated_anomalies_train, len(binary_cols)))
            anoms_synthetic_training_numeric[:, binary_cols] = bernoulli_vars

        # categorical variables
        ohe = []
        for i, num_vals in enumerate(one_hot_col_len):
            np.random.seed(seed_anom_generation + 321 + i)
            feat = np.random.randint(num_vals, size=num_generated_anomalies_train)
            # convert to one-hot array
            ohe_feat = np.zeros((feat.size, num_vals))
            ohe_feat[np.arange(feat.size), feat] = 1
            ohe.append(ohe_feat)
        anoms_synthetic_training = np.hstack(ohe + [anoms_synthetic_training_numeric])
    else:
        anoms_synthetic_training = None
        num_generated_anomalies_training = None

    anom_train, anom_val = split_synthetic_anom(num_generated_anomalies_training, anoms_synthetic_training)

    x_train, y_train = concat_real_synthetic(x_train_real, y_train_real, anom_train)
    x_val, y_val = concat_real_synthetic(x_val_real, y_val_real, anom_val)

    return x_train, y_train, x_val, y_val


def split_synthetic_anom(num_train, anoms_synthetic):
    if num_train is None or anoms_synthetic is None:
        return anoms_synthetic, None
    anom_train = anoms_synthetic[:num_train]
    anom_val = anoms_synthetic[num_train:]
    return anom_train, anom_val


def concat_real_synthetic(x, y, synthetic):
    """
    Concatenate synthetic anomalies to real data
    Args:
        x: real data features
        y: real data labels
        synthetic: synthetic data features

    Returns: tuple of (features, labels) where 0 is anomaly and 1 is normal

    """
    if x is None:
        return None, None
    if synthetic is None:
        return x, y
    features = np.vstack((x, synthetic))
    labels = np.hstack((y, np.zeros(len(synthetic))))
    return features, labels


def get_datasets_by_label(x, y, torch_dataset=True, base_class=0):
    classes = np.unique(y)
    classes.sort()
    datasets = []
    for cls in classes:
        indices = (y == cls)
        features = x[indices]
        num = np.sum(indices)
        if cls == base_class:
            label = np.ones(num)
        else:
            label = np.zeros(num)
        if torch_dataset:
            dataset = TensorDataset(torch.Tensor(features), torch.Tensor(label))
        else:
            dataset = (features, label)
        datasets.append(dataset)

    return datasets


def get_normal_label(pos_label=1, normal_is_positive=False):

    return pos_label if normal_is_positive else (1 - pos_label)


def get_in_range_date(x, y, lower=0, upper=1, print_drop_count=False):
    valid_indices = np.logical_or(lower <= x, x <= upper).any(axis=1)
    if print_drop_count:
        print(f"Dropped {len(x) - len(valid_indices)} rows")
    return x[valid_indices], y[valid_indices]


def get_data(dataset_name, **kwargs):
    raw_label_col = "attack"
    if dataset_name == "kdd":
        # KDD
        from Utils.kdd import map_attack_kdd, columns, attack_labels
        path_train = '../data/kdd/KDDTrain+.txt'
        path_test = '../data/kdd/KDDTest+.txt'

        features_to_encode = ['protocol_type', 'service', 'flag']
        df = get_df(path_train, columns=columns, drop=False)
        test_df = get_df(path_test, columns=columns, drop=False)

        new_attacks = [1, 2, 3, 4]
        test_classes = [0, 1, 2, 3, 4]

        # get numeric features, we won't worry about encoding these at this point
        # numeric_features = ['duration', 'src_bytes', 'dst_bytes']
        # Use all features
        numeric_features = sorted(list(set(df.columns[:-5]) - set(features_to_encode)))
        map_attack = map_attack_kdd
        normal_label = "normal"

    elif dataset_name == "kitsune":
        # Kitsune
        data_dir = '../data/kitsune+network+attack+dataset/'
        # CHOOSE
        mirai = kwargs.get("mirai", False)
        #########################
        if mirai:
            dataset_name += "_mirai"
            att_dir = data_dir + "mirai/"
            x_folder = att_dir + "Mirai_dataset.csv/Mirai_dataset.csv"
            y_folder = att_dir + "Mirai_labels.csv/mirai_labels.csv"
            # data = pd.read_csv(x_folder, header=None)
            data = get_df(x_folder, columns=["index"] + list(range(115)), drop=["index"])
            labels = pd.read_csv(y_folder).iloc[:, -1].rename("attack")

            # Train/Test split
            training_num = 10000
            df = data.iloc[:training_num]
            df['attack'] = 0
            test_df = data.iloc[training_num:]
            test_df['attack'] = labels.iloc[training_num:]

            normal_label = 0
            attack_labels = ['Normal', "Mirai"]

            new_attacks = [1]
            test_classes = [0, 1]

            def map_attack(i):
                return i

        else:
            custom = kwargs.get("custom", "")
            # custom = "syn_dos"
            if len(custom) > 0:
                dataset_name += "_" + custom
                att_dir = data_dir + custom + "/"
                data_folder, label_folder = [os.path.join(att_dir, pth) for pth in os.listdir(att_dir) if
                                             pth.endswith(".csv")]
                data_csv_file = os.listdir(data_folder)[0]
                x_folder = os.path.join(data_folder, data_csv_file)
                label_csv_file = os.listdir(label_folder)[0]
                y_folder = os.path.join(label_folder, label_csv_file)
                data = get_df(x_folder, columns=None, drop=False)
                labels = pd.read_csv(y_folder, usecols=["x"])

                # Train/Test split
                training_num = 1000000
                df = data.iloc[:training_num]
                df['attack'] = 0
                test_df = data.iloc[training_num:]
                test_df['attack'] = labels.iloc[training_num:]

                normal_label = 0
                attack_labels = ['Normal', custom]

                new_attacks = [1]
                test_classes = [0, 1]

                def map_attack(i):
                    return i
            else:
                train_path = data_dir + "train_normal_all.csv"
                test_path = data_dir + "test_all.csv"
                # df = pd.read_csv(train_path, header=None, nrows=1000).drop_duplicates().reset_index(drop=True)
                # test_df = pd.read_csv(test_path, header=None, skiprows=10000, nrows=500000).drop_duplicates().reset_index(drop=True)
                df = get_df(train_path, columns=None, drop=False)
                test_df = get_df(test_path, columns=None, drop=False)

                normal_label = 0
                attack_labels = ['Normal', 'Active Wiretap', 'ARP MitM', 'Fuzzing', 'OS Scan', 'SSDP Flood',
                                 'SSL Renegotiation', 'SYN_DoS', 'Video Injection']
                # attack_labels = ['Normal', 'active_wiretap', 'arp_mitm', 'fuzzing', 'os_scan', 'ssdp_flood',
                #                  'ssl_renegotiation', 'syn_dos', 'video_injection']
                raw_label_col = 115

                new_attacks = [1, 2, 3, 4, 5, 6, 7, 8]
                test_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

                def map_attack(i):
                    if i <= 3:
                        return i
                    # attack 4 (mirai) is in a separate dataset
                    return i - 1


        features_to_encode = []
        n_dim = 115
        numeric_features = list(range(n_dim))

        # attack_labels = ['Normal', att.replace("_", " ").title()]

        # new_attacks = [1]
        # test_classes = [0]

    else:
        raise ValueError("Dataset not supported")

    return (df, test_df, features_to_encode, numeric_features, normal_label, attack_labels, new_attacks, test_classes,
            raw_label_col, map_attack)


def np_to_dataloader(x_training_list, y_training_list, x_test, y_test, batch_size):
    """
    convert np arrays for training/validation and testing to dataloader
    Args:
        x_training_list: [x_train, x_val] or [x_train]
        y_training_list: [y_train, y_val] or [y_train]
        x_test:
        y_test:
        batch_size:

    Returns: training_dataloaders, testing_datasets, test_loader

    """
    training_dataloaders = [
        get_dataloader(x, y, batch_size=batch_size) for x, y in zip(x_training_list, y_training_list)]
    testing_datasets = get_datasets_by_label(x_test, y_test)
    test_loader = get_dataloader(x_test, (y_test == 0), batch_size=batch_size)
    return training_dataloaders, testing_datasets, test_loader
