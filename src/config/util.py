import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import OrderedDict, Union

import numpy as np
import torch
from path import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
LOG_DIR = PROJECT_DIR / "logs"
TEMP_DIR = PROJECT_DIR / "temp"
DATA_DIR = PROJECT_DIR / "data"


def get_mnist_loaders(client_num_per_round):
    fix_random_seed(seed=1)
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    # normalize & reshape
    X_train = train_ds.data.float().unsqueeze(1) / 255.0
    X_test  = test_ds.data .float().unsqueeze(1) / 255.0
    y_train = train_ds.targets
    y_test  = test_ds.targets 

    # global loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test ), batch_size=1024, shuffle=False)

    # split into client loaders
    perm  = torch.randperm(len(X_train))
    chunk = len(X_train) // client_num_per_round
    idxs = [perm[i*chunk:(i+1)*chunk] for i in range(client_num_per_round)]
    idxs[-1] = perm[(client_num_per_round-1)*chunk:]


    client_loaders = [
        DataLoader(TensorDataset(X_train[i], y_train[i]), batch_size=64, shuffle=True)
        for i in idxs
    ]

    return train_loader, test_loader, client_loaders


def fix_random_seed(seed: int) -> None:
    torch.cuda.empty_cache()
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_epochs", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--local_lr", type=float, default=1e-2)
    parser.add_argument("--verbose_gap", type=int, default=5)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "emnist", "fmnist"],
        default="mnist",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--client_num_per_round", type=int, default=5)
    parser.add_argument("--save_period", type=int, default=5)
    parser.add_argument(
        "--dp_sigma",
        type=float,
        default=0.1,
        help="std of Gaussian noise for DP",
    )
    parser.add_argument(
        "--clip_bound",
        type=float,
        default=100.0,
        help="clipping bound for gradient",
    )
    parser.add_argument(
        "--jse",
        type=bool,
        default=False,
        help="apply james-stein estimator",
    )
    return parser.parse_args()


def add_dp_noise(
    pseudo_gradient: OrderedDict[str, torch.Tensor], sigma: float, seed: int, clip_norm: float
) -> OrderedDict[str, torch.Tensor]:
    torch.manual_seed(seed)
    noisy_pseudo_gradient = OrderedDict()
    
    # First, compute the total norm of all gradients
    total_norm = 0.0
    for name, param in pseudo_gradient.items():
        total_norm += param.norm().item() ** 2
    total_norm = total_norm ** 0.5
    
    # Compute clipping coefficient
    clip_coef = min(1.0, clip_norm / (total_norm + 1e-6))
    
    for name, param in pseudo_gradient.items():
        # Clip the gradient
        # print('param', param)
        clipped_param = param * clip_coef
        # Ensure noise is on the same device as the parameter
        noise = torch.randn_like(clipped_param, device=clipped_param.device) * sigma
        # print('clipped_param', clipped_param)
        # print('randn_like', torch.randn_like(clipped_param, device=clipped_param.device) )
        noisy_pseudo_gradient[name] = clipped_param + noise
        # print('sigma', sigma,'noise', noise)
        # print('noisy_pseudo_gradient[name]', noisy_pseudo_gradient[name])
    return noisy_pseudo_gradient


def add_dp_noise_only(
    pseudo_gradient: OrderedDict[str, torch.Tensor], sigma: float
) -> OrderedDict[str, torch.Tensor]:
    """
    Add differential privacy noise to pseudo_gradient without clipping.
    Clipping should be done during local training.
    """
    noisy_pseudo_gradient = OrderedDict()
    
    for name, param in pseudo_gradient.items():
        # Add Gaussian noise only (no clipping)
        noise = torch.randn_like(param, device=param.device) * sigma
        noisy_pseudo_gradient[name] = param + noise

        # snr = param.norm() / noise.norm()
        # avg_noise = noise.abs().mean().item()
        # avg_param = param.abs().mean().item()
        # ratio = avg_noise / (avg_param + 1e-8)

        # print(f"{name}: SNR={snr:.2f}, noise/param ratio={ratio:.2f}")
    
    return noisy_pseudo_gradient

def apply_jse(noisy_term, sigma, d):
    """
    Apply James-Stein estimator to noisy gradients for local DP.
    Args:
        sigma: std of Gaussian noise for DP, args.dp_sigma
        d: sum(p.numel() for p in global_model.parameters())
        noisy_term: Parameter gradient tensor
    Returns:
        Shrunk gradient tensor
    """
    numerator = (d - 2) * (sigma ** 2)
    # print("d", d)
    denominator = noisy_term.pow(2).sum().item()
    # print('>>> denominator', denominator)
    shrinkage_factor = 1.0 - numerator / denominator
    # print('>>> shrinkage_factor', shrinkage_factor)
    shrinkage_factor = max(shrinkage_factor, 0.0001)
    return noisy_term * shrinkage_factor
