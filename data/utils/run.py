import time
from path import Path

_CURRENT_DIR = Path(__file__).parent.abspath()
import sys

sys.path.append(_CURRENT_DIR)
sys.path.append(_CURRENT_DIR.parent)
import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST, FashionMNIST

from constants import MEAN, STD
from partition import randomly_assign_classes, dirichlet_noniid_partition
from utils.dataset import CIFARDataset, MNISTDataset
from util import set_seed

DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "emnist": (EMNIST, MNISTDataset),
    "fmnist": (FashionMNIST, MNISTDataset),
    "cifar10": (CIFAR10, CIFARDataset),
    "cifar100": (CIFAR100, CIFARDataset),
}


def main(args):
    _DATASET_ROOT = (
        Path(args.root).abspath() / args.dataset
        if args.root is not None
        else _CURRENT_DIR.parent / args.dataset
    )
    _PICKLES_DIR = _CURRENT_DIR.parent / args.dataset / "pickles"

    set_seed(args.seed)

    classes_map = None
    transform = transforms.Compose(
        [
            transforms.Normalize(MEAN[args.dataset], STD[args.dataset]),
        ]
    )
    target_transform = None

    if not os.path.isdir(_DATASET_ROOT):
        os.makedirs(_DATASET_ROOT)
    if os.path.isdir(_PICKLES_DIR):
        os.system(f"rm -rf {_PICKLES_DIR}")
    os.makedirs(_PICKLES_DIR, exist_ok=True)

    client_num_in_total = args.client_num_in_total
    client_num_in_total = args.client_num_in_total
    # TODO: ori_dataset is image dataset, target_dataset is tensor dataset?
    # TODO: ori_dataset is image dataset, target_dataset is tensor dataset?
    ori_dataset, target_dataset = DATASET[args.dataset]
    if args.dataset == "emnist":
        trainset = ori_dataset(
            _DATASET_ROOT,
            train=True,
            download=True,
            split=args.emnist_split,
            transform=transforms.ToTensor(),
        )
        testset = ori_dataset(
            _DATASET_ROOT,
            train=False,
            split=args.emnist_split,
            transform=transforms.ToTensor(),
        )
    else:
        trainset = ori_dataset(
            _DATASET_ROOT,
            train=True,
            download=True,
        )
        testset = ori_dataset(
            _DATASET_ROOT,
            train=False,
        )
    # concat_datasets = [trainset, testset]
    concat_datasets = [trainset]
    
    # Create proper test datasets for each client
    test_datasets = []
    samples_per_client = len(testset) // client_num_in_total
    
    for i in range(client_num_in_total):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        
        # Get both data and targets for this split
        test_data = testset.data[start_idx:end_idx]
        test_targets = testset.targets[start_idx:end_idx]
        
        # Convert to tensor if needed
        if not isinstance(test_data, torch.Tensor):
            test_data = transforms.ToTensor()(test_data)
        test_data = test_data.float()
        
        # Apply normalization
        test_data = transforms.Normalize(MEAN[args.dataset], STD[args.dataset])(test_data)
        
        # Create dataset
        full_test_dataset = target_dataset(
            data=test_data,
            targets=test_targets,
            transform=transform,
            target_transform=target_transform
        )
        
        # Wrap in Subset to match train/val type
        indices = list(range(len(full_test_dataset)))
        test_datasets.append(torch.utils.data.Subset(full_test_dataset, indices))

    if args.alpha > 0:  # NOTE: Dirichlet(alpha)
        all_datasets, stats = dirichlet_noniid_partition(
            train_data_raw=trainset,
            target_dataset=target_dataset,
            num_clients=client_num_in_total,
            transform=transform,
            target_transform=target_transform,
        )
    else:  # NOTE: sort and partition
        classes = len(ori_dataset.classes) if args.classes <= 0 else args.classes
        all_datasets, stats = randomly_assign_classes(
            ori_datasets=concat_datasets,
            target_dataset=target_dataset,
            num_clients=client_num_in_total,
            num_classes=classes,
            transform=transform,
            target_transform=target_transform,
        )
    # Print sizes for all clients
    print("\nDataset sizes per client:")
    for i, dataset in enumerate(all_datasets):
        print(f"Client {i}: {len(dataset)} samples")
    for subset_id, client_id in enumerate(
        range(0, len(all_datasets), args.client_num_in_each_pickles)
    ):
        subset = []
        for idx, dataset in enumerate(all_datasets[
            client_id : client_id + args.client_num_in_each_pickles
        ]):
            current_client_id = client_id + idx
            num_val_samples = int(len(dataset) * args.valset_ratio)
            num_train_samples = len(dataset) - num_val_samples
            train = torch.utils.data.Subset(dataset, list(range(len(dataset))))
            val = None

            num_test_samples = len(test_datasets[current_client_id])
            print("-"*20, "Client", current_client_id, "-"*20)
            print(f"  - Train samples: {len(train)}")
            print(f"  - Val: {val}")
            print(f"  - Test samples: {num_test_samples}")
            subset.append({
                "train": train, 
                "val": val, 
                "test": test_datasets[current_client_id]
                })
        with open(_PICKLES_DIR / str(subset_id) + ".pkl", "wb") as f:
            pickle.dump(subset, f)

    # Create normalized training dataset for global model evaluation
    train_data = trainset.data
    if not isinstance(train_data, torch.Tensor):
        if isinstance(train_data, np.ndarray):
            train_data = torch.from_numpy(train_data)
        else:
            train_data = transforms.ToTensor()(train_data)
    train_data = train_data.float()
    
    # Handle different dataset formats
    if train_data.dim() == 3:  # Add channel dimension for grayscale images
        train_data = train_data.unsqueeze(1)
    elif train_data.dim() == 4:  # Already has channel dimension
        pass
    else:
        raise ValueError(f"Unexpected data dimension: {train_data.dim()}")
        
    train_data = transforms.Normalize(MEAN[args.dataset], STD[args.dataset])(train_data)
    
    # Convert targets to tensor if needed
    targets = trainset.targets
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    
    full_train_dataset = target_dataset(
        data=train_data,
        targets=targets,
        transform=transform,
        target_transform=target_transform
    )
    
    # Save the normalized training dataset for server use
    with open(_PICKLES_DIR / "full_trainset.pkl", "wb") as f:
        pickle.dump(full_train_dataset, f)

    # Create normalized test dataset for global model evaluation
    test_data = testset.data
    if not isinstance(test_data, torch.Tensor):
        if isinstance(test_data, np.ndarray):
            test_data = torch.from_numpy(test_data)
        else:
            test_data = transforms.ToTensor()(test_data)
    test_data = test_data.float()
    
    # Handle different dataset formats
    if test_data.dim() == 3:  # Add channel dimension for grayscale images
        test_data = test_data.unsqueeze(1)
    elif test_data.dim() == 4:  # Already has channel dimension
        pass
    else:
        raise ValueError(f"Unexpected data dimension: {test_data.dim()}")
        
    test_data = transforms.Normalize(MEAN[args.dataset], STD[args.dataset])(test_data)
    
    # Convert targets to tensor if needed
    test_targets = testset.targets
    if not isinstance(test_targets, torch.Tensor):
        test_targets = torch.tensor(test_targets)
    
    full_test_dataset = target_dataset(
        data=test_data,
        targets=test_targets,
        transform=transform,
        target_transform=target_transform
    )
    
    # Save the normalized test dataset for server use
    with open(_PICKLES_DIR / "full_testset.pkl", "wb") as f:
        pickle.dump(full_test_dataset, f)

    # save stats
    if args.type == "user":
        train_clients_num = int(client_num_in_total * args.fraction)
        clients_4_train = [i for i in range(train_clients_num)]
        clients_4_test = [i for i in range(train_clients_num, client_num_in_total)]

        with open(_PICKLES_DIR / "seperation.pkl", "wb") as f:
            pickle.dump(
                {
                    "train": clients_4_train,
                    "test": clients_4_test,
                    "total": client_num_in_total,
                },
                f,
            )

        train_clients_stats = dict(
            zip(clients_4_train, list(stats.values())[:train_clients_num])
        )
        test_clients_stats = dict(
            zip(
                clients_4_test,
                list(stats.values())[train_clients_num:],
            )
        )

        with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
            json.dump({"train": train_clients_stats, "test": test_clients_stats}, f)

    else:  # NOTE: "sample"  save stats
        client_id_indices = [i for i in range(client_num_in_total)]
        with open(_PICKLES_DIR / "seperation.pkl", "wb") as f:
            pickle.dump(
                {
                    "id": client_id_indices,
                    "total": client_num_in_total,
                },
                f,
            )
        with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
            json.dump(stats, f)

    args.root = (
        Path(args.root).abspath()
        if str(_DATASET_ROOT) != str(_CURRENT_DIR.parent / args.dataset)
        else None
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "emnist",
            "fmnist",
        ],
        default="mnist",
    )

    parser.add_argument("--client_num_in_total", type=int, default=5)
    parser.add_argument(
        "--fraction", type=float, default=0.9, help="Propotion of train clients"
    )
    parser.add_argument("--valset_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--classes",
        type=int,
        default=-1,
        help="Num of classes that one client's data belong to.",
    )
    parser.add_argument("--seed", type=int, default=42)
    ################# Dirichlet distribution only #################
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="Only for controling data hetero degree while performing Dirichlet partition.",
    )
    ###############################################################

    ################# For EMNIST only #####################
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
    )
    #######################################################
    parser.add_argument(
        "--type", type=str, choices=["sample", "user"], default="sample"
    )
    parser.add_argument("--client_num_in_each_pickles", type=int, default=10)
    parser.add_argument("--root", type=str, default="/root/repos/python/mine/datasets")
    args = parser.parse_args()
    main(args)
    args_dict = dict(args._get_kwargs())
    with open(_CURRENT_DIR.parent / "args.json", "w") as f:
        json.dump(args_dict, f)
