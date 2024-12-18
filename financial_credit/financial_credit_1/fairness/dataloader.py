import torch
import numpy as np
import torch.utils.data as data
import pandas as pd
import json
from collections import defaultdict

# Constants
UCI_ADULT_PATH = "data/datasets/uci_adult/"
COMPAS_PATH = "data/datasets/compas/"
LAW_SCHOOL_PATH = "data/datasets/law_school/"

class loadDataset(data.Dataset):
    def __init__(self, dataset, train_or_test, embedding_size=None):
        """
        Creates a PyTorch dataloader object.

        Args:
            dataset: name of dataset (uci_adult, compas, law_school).
            train_or_test: wheter to load train or test set (string).
            embedding_size: embedding size to use for categorical data.
        """

        # Read the dataset, and its characteristics.
        self.dataframe, self.dataset_stats = get_dataset_stats(
            dataset, train_or_test
        )

        # Save the characteristics of this dataset in variables. 
        # E.g. what is the target variable, protected features, etc.
        self.mean_std = self.dataset_stats["mean_std"]
        self.vocabulary = self.dataset_stats["vocabulary"]
        self.target_column_name = self.dataset_stats["target_column_name"]
        self.target_column_positive_value = self.dataset_stats[
            "target_column_positive_value"
        ]
        # self.sensitive_column_names = self.dataset_stats["sensitive_column_names"]
        # self.sensitive_column_values = self.dataset_stats["sensitive_column_values"]
        # self.sensitive_column_names = []
        self.sensitive_column_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
        self.sensitive_column_values = []
        self.embedding_size = embedding_size

        # Ensures self.dataframe has the correct dtype for all columns.
        self.prepare_dataframe()

        # Binarize target variables.
        self.binarize()

        # Normalize numerical data to zero mean and variance
        self.normalize()

        if embedding_size:
            # Calculate the embedding sizes.
            self.calculate_embedding()

        # Extract all protected subgroups from the dataset. 
        # E.g. [male], [black], but also [black male]
        # self.set_subgroups()

        # Transforms all data to on-hot-encoded for training.
        self.stack_data()

    def prepare_dataframe(self):
        """
        Ensures all columns have the correct dtype.
        """
        # Rename dataframe columns and replace empty string with 'unk'
        self.dataframe.columns = self.dataset_stats["feature_names"]
        self.dataframe.fillna("unk", inplace=True)

        # Change dtype to categorical for all category columns
        for category in self.vocabulary.keys():
            self.dataframe[category] = self.dataframe[category].astype(
                "category"
            )

    def binarize(self):
        """
        Ensures target data and protected features are binary.
        """
        # Binarize target variables.
        self.dataframe[self.target_column_name] = self.dataframe[
            self.target_column_name
        ].astype("category")
        self.dataframe[self.target_column_name] = (
            self.dataframe[self.target_column_name] 
            == self.target_column_positive_value) * 1
        self.target_data = torch.Tensor(
            self.dataframe[self.target_column_name].values
        )

        # Binarize protected features. 
        for sensitive_column_name, sensitive_column_value in zip(
            self.sensitive_column_names, self.sensitive_column_values
        ):
            self.dataframe[sensitive_column_name] = (
                self.dataframe[sensitive_column_name] == sensitive_column_value
            ) * 1
        # self.protected_data = torch.Tensor(
        #     self.dataframe[self.sensitive_column_names].values
        # )

    def normalize(self):
        """
        Ensures numerical data has zero mean and variance.
        """
        for key, value in self.mean_std.items():
            mean = value[0]
            std = value[1]
            self.dataframe[key] = (self.dataframe[key] - mean) / std

    def calculate_embedding(self):
        """
        Calculates the embedding size for categorical data.
        """
        self.categorical_embedding_sizes = [
            (len(vocab) + 1, self.embedding_size)
            for cat, vocab in self.vocabulary.items()
            if cat not in self.sensitive_column_names
            and cat != self.target_column_name
        ]

    def set_subgroups(self):
        """
        Use the cartesian product to get subgroups of protected groups.
        for example the subgroups: [male] and [black] but also [black male].
        """
        opt = self.protected_data.unique().numpy()
        combinations = np.transpose([np.tile(opt, len(opt)),
             np.repeat(opt, len(opt))])
        subgroups = [np.where((self.dataframe
            [self.sensitive_column_names[0]] == comb[0]) 
            & (self.dataframe[self.sensitive_column_names[1]] == comb[1]), 1, 0) 
            for idx, comb in enumerate(combinations)]

        # Add the subgroups  [male], [female] [white], [black],
        for col in self.sensitive_column_names:
            for option in opt: 
                subgroups.append(np.where((self.dataframe[col] == option), 1, 0))
        self.subgroups = pd.DataFrame(subgroups).transpose()

        # Get the minority subgroup (subgroup that is least supported).
        subgroup_counts = [list(self.subgroups[c].value_counts()) 
            for c in self.subgroups.columns]
        self.subgroup_minority = np.argmin(np.array(subgroup_counts), axis=0)[1]

        # Get the indexes of the dataframe rows that correspond to each subgroup.
        subgroup_indexes = []
        for col in range(len(self.subgroups.columns)):
            subgroup_indexes.append(self.subgroups.index
            [self.subgroups[col] == 1].tolist())
        self.subgroup_indexes = subgroup_indexes

    def stack_data(self):
        """
        Change categorical data to one-hot encoded tensors.
        """
        one_hot_encoded = [
            self.dataframe[feature].cat.codes.values
            for feature in self.vocabulary.keys()
            if feature not in self.sensitive_column_names
            and feature != self.target_column_name
        ]
        self.categorical_data = torch.tensor(
            np.stack(one_hot_encoded, 1), dtype=torch.int64
        )

        # Stack numerical data into tensors.
        numerical_data = np.stack(
            [self.dataframe[col].values for col in self.mean_std.keys()], 1
        )
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float)

    def __getitem__(self, idx):
        """
        Returns one data instance from dataframe.
        """
        categorical_data = self.categorical_data[idx]
        numerical_data = self.numerical_data[idx]
        target_data = self.target_data[idx].reshape(-1).float()

        return categorical_data, numerical_data, target_data

    def get_split(self, idx):
        """
        Returns a set of data instances, for instance test set.
        """
        categorical_data = self.categorical_data[idx]
        numerical_data = self.numerical_data[idx]
        target_data = self.target_data[idx].reshape(-1).float()
        return IterableDataset(categorical_data, numerical_data, target_data)

    def __len__(self):
        return len(self.dataframe)

    @property
    def vocab_size(self):
        return self._vocab_size


class IterableDataset:
    def __init__(self, cat, num, target):
        """
        An iterable dataset that can be passed to pytorches DataLoader.
        """
        self.categorical_data = cat
        self.numerical_data = num
        self.target_data = target

    def __getitem__(self, idx):
        categorical_data = self.categorical_data[idx]
        numerical_data = self.numerical_data[idx]
        target_data = self.target_data[idx].reshape(-1).float()

        return categorical_data, numerical_data, target_data

    def __len__(self):
        return len(self.target_data)


def get_dataset_stats(dataset, train_or_test):
    """
    Returns input feature values for each dataset.

    args:
        dataset: the dataset to load (compas, law_school or uci_adult)
        train_or_test: string, specifies either train or test set
    """
    if dataset == "compas":
        data_path = COMPAS_PATH
    elif dataset == "law_school":
        data_path = LAW_SCHOOL_PATH
    elif dataset == "uci_adult":
        data_path = UCI_ADULT_PATH

    # Read the dataframe.
    dataframe = pd.read_csv(
        data_path + train_or_test + ".csv", header=None
    )

    # Read json file to determine which columns of data to use.
    with open(data_path + "dataset_stats.json") as f:
        dataset_stats = json.load(f)

    # Read json files to distinguish between categorical/numerical data
    with open(data_path + "vocabulary.json") as f:
        dataset_stats["vocabulary"] = json.load(f)
    with open(data_path + "mean_std.json") as f:
        dataset_stats["mean_std"] = json.load(f)

    return dataframe, dataset_stats


class TensorBoardLogger(object):
    def __init__(self, summary_writer, avg_window=5, name=None):
        """
        Class that summarizes some logging code for TensorBoard.
        Open with "tensorboard --logdir logs/" in terminal.
        
        args:
            summary_writer: Summary Writer object from torch.utils.tensorboard.
            avg_window: How often to update the logger. 
            name: Tab name in TensorBoard's scalars.
        """
        self.summary_writer = summary_writer
        if name is None:
            self.name = ""
        else:
            self.name = name + "/"

        self.value_dict = defaultdict(lambda: 0)
        self.steps = defaultdict(lambda: 0)
        self.global_step = 0
        self.avg_window = avg_window

    def add_values(self, log_dict):
        """
        Function for adding a dictionary of logging values to this logger.

        args:
            log_dict:Dictionary of string to Tensor with the values to plot.
        """
        self.global_step += 1

        for key, val in log_dict.items():
            # Detatch if necissary
            if torch.is_tensor(val):
                val = val.detach().cpu()
            self.value_dict[key] += val
            self.steps[key] += 1

            # Plot to TensorBoard every avg_window steps
            if self.steps[key] >= self.avg_window:
                avg_val = self.value_dict[key] / self.steps[key]
                self.summary_writer.add_scalar(
                    self.name + key, avg_val, global_step=self.global_step
                )
                self.value_dict[key] = 0
                self.steps[key] = 0
