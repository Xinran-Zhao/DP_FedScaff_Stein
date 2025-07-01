from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def dirichlet_distribution(
    trainset: Dataset,
    ori_dataset: List[Dataset],
    target_dataset: Dataset,
    num_clients: int,
    alpha: float,
    transform=None,
    target_transform=None,
) -> Tuple[List[Dataset], Dict]:
    NUM_CLASS = len(ori_dataset[0].classes)
    MIN_SIZE = 0
    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    stats = {}
    targets_numpy = np.concatenate(
        [ds.targets for ds in ori_dataset], axis=0, dtype=np.int64
    )
    # If ori_dataset has only one element, concatenation is unnecessary
    # if len(ori_dataset) == 1:
    data_numpy = np.concatenate(
        [ds.data for ds in ori_dataset], axis=0, dtype=np.float32
    )
    print("data_numpy type:", type(data_numpy))
    print("data_numpy shape:", data_numpy.shape)
    print("data_numpy_by_trainset type:", type(data_numpy_by_trainset))
    print("data_numpy_by_trainset shape:", data_numpy_by_trainset.shape)
    idx = [np.where(targets_numpy == i)[0] for i in range(NUM_CLASS)]

    while MIN_SIZE < 10: # ensures each client gets at least 10 samples
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(NUM_CLASS):
            np.random.shuffle(idx[k])
            distributions = np.random.dirichlet(np.repeat(alpha, num_clients))
            distributions = np.array(
                [
                    p * (len(idx_j) < len(targets_numpy) / num_clients)
                    for p, idx_j in zip(distributions, idx_batch)
                ]
            )
            distributions = distributions / distributions.sum()
            distributions = (np.cumsum(distributions) * len(idx[k])).astype(int)[:-1]
            idx_batch = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(idx_batch, np.split(idx[k], distributions))
            ]
            MIN_SIZE = min([len(idx_j) for idx_j in idx_batch])

        for i in range(num_clients):
            stats[i] = {"x": None, "y": None}
            np.random.shuffle(idx_batch[i])
            X[i] = data_numpy[idx_batch[i]]
            Y[i] = targets_numpy[idx_batch[i]]
            stats[i]["x"] = len(X[i])
            stats[i]["y"] = Counter(Y[i].tolist())

    print("X type:", type(X))
    # Print shapes of individual arrays in X to debug
    print("X element shapes:", [x.shape for x in X])
    print("Y type:", type(Y))
    print("Y element shapes:", [y.shape for y in Y])

    datasets = [
        target_dataset(
            data=X[j],
            targets=Y[j],
            transform=transform,
            target_transform=target_transform,
        )
        for j in range(num_clients)
    ]
    print('type of datasets:', type(datasets))
    print(f"Type of elements in train_datasets: {type(datasets[0])}")
    ohygod
    return datasets, stats

def dirichlet_noiid_partition(
        train_data_raw: Dataset, 
        num_clients: int, 
        target_dataset: Dataset,
        seed=42, 
        alpha1=100, 
        alpha2=0.1, 
        transform=None,
        target_transform=None,
        ) -> Tuple[List[Dataset], Dict]:
    data_numpy = train_data_raw.data.numpy().astype(np.float32)
    targets_numpy = train_data_raw.targets.numpy().astype(np.float32)
    np.random.seed(seed)
    # Split the dataset by label
    labels = []
    label_indices = {} # key is label, value is list of indices
    for idx, (_, label) in enumerate(train_data_raw):
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()
    # Shuffle the indices for different label
    for label in labels:
        np.random.shuffle(label_indices[label])

    p1 = [1 / num_clients for _ in range(num_clients)]      # prior distribution for each client's number of elements
    p2 = [len(label_indices[label]) for label in labels]
    p2 = [p / sum(p2) for p in p2]
    q1 = [alpha1 * i for i in p1]
    q2 = [alpha2 * i for i in p2]

    weights = np.random.dirichlet(q1) # the total number of elements for each client
    individuals = np.random.dirichlet(q2, num_clients) # the number of elements from each class for each client

    number_samples_classwise = [len(label_indices[label]) for label in labels]

    normalized_portions = np.zeros(individuals.shape)
    for i in range(num_clients):
        for j in range(len(number_samples_classwise)):
            normalized_portions[i][j] = weights[i] * individuals[i][j] / np.dot(weights, individuals.transpose()[j])

    res = np.multiply(np.array([number_samples_classwise] * num_clients), normalized_portions).transpose()

    for i in range(len(number_samples_classwise)):
        total = 0
        for j in range(num_clients - 1):
            res[i][j] = int(res[i][j])
            total += res[i][j]
        res[i][num_clients - 1] = number_samples_classwise[i] - total
     
    # number of elements from each class for each client. shape: (num_clients, num_classes)
    num_elements = np.array(res.transpose(), dtype=np.int32)
    sum_elements = np.cumsum(num_elements, axis=0)
    print("the num_elements is: \n", num_elements)

    stats = {}  # Initialize stats dictionary
    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    idx_batch = [[] for _ in range(num_clients)]
    # get real idx_batch
    for i in range(num_clients):
        for j in range(len(labels)):
            start = 0 if i == 0 else sum_elements[i-1][j]
            end = sum_elements[i][j]
            for idx in label_indices[labels[j]][start:end]:
                idx_batch[i].append(idx)
    
    for i in range(num_clients):
        stats[i] = {"x": None, "y": None}
        np.random.shuffle(idx_batch[i])
        X[i] = data_numpy[idx_batch[i]]
        Y[i] = targets_numpy[idx_batch[i]]
        stats[i]["x"] = len(X[i])
        stats[i]["y"] = Counter(Y[i].tolist())

    datasets = [
        target_dataset(
            data=X[j],
            targets=Y[j],
            transform=transform,
            target_transform=target_transform,
        )
        for j in range(num_clients)
    ]
    return datasets, stats
