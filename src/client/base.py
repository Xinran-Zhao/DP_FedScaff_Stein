from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import torch
import numpy as np
from path import Path
from rich.console import Console
from torch.utils.data import Subset, DataLoader, TensorDataset

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from data.utils.util import get_dataset, set_seed
from src.config.util import add_dp_noise, add_dp_noise_only, apply_jse, get_mnist_loaders, fix_random_seed


class ClientBase:
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        batch_size: int,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
        dp_sigma: float,
        clip_bound: float,
        jse: bool,
    ):
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        fix_random_seed(1)
        self.client_id: int = None
        self.valset: Subset = None
        self.trainset: Subset = None
        self.testset: Subset = None
        # initialize model
        self.model: torch.nn.Module = deepcopy(backbone).to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=local_lr
        )
        
        # Print initial model parameters for debugging
        print(f"=== CLIENT {self.client_id} INITIAL MODEL ===")
        initial_params = torch.cat([p.detach().clone().flatten() for p in self.model.parameters()])
        print(f"Initial params norm: {torch.norm(initial_params).item():.6f}")
        print(f"Initial params mean: {initial_params.mean().item():.6f}")
        print(f"Initial params std: {initial_params.std().item():.6f}")
        print(f"First 10 params: {initial_params[:10]}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.dp_sigma = dp_sigma
        self.clip_bound = clip_bound
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logger
        self.untrainable_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.jse = jse
        
    @torch.no_grad()
    def evaluate(self, use_valset=False):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = 0
        correct = 0
        dataloader = DataLoader(self.valset if use_valset else self.trainset, 32)
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss += criterion(logits, y)
            pred = torch.softmax(logits, -1).argmax(-1)
            correct += (pred == y).int().sum()
        return loss.item(), correct.item()

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        epoch: int,
        evaluate=True,
        verbose=False,
        use_valset=False, # use trainset to evaluate instead of valset
    ) -> Tuple[List[torch.Tensor], int, OrderedDict[str, torch.Tensor]]:
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        parms_before_training = deepcopy(self.model.state_dict())
        res, stats = self._log_while_training(evaluate, verbose, use_valset)()
        params_after_training = self.model.state_dict()
        pseudo_grad = OrderedDict()
        for key in parms_before_training.keys():
            pseudo_grad[key] = params_after_training[key] - parms_before_training[key]

        if self.dp_sigma > 0:
            # Need to add args.seed and current round as parameters to train() method
            # Add to method signature:
            # def train(self, client_id: int, model_params: OrderedDict[str, torch.Tensor], 
            #          epoch: int, seed: int, round: int, evaluate=True, verbose=False, use_valset=False)
            torch.manual_seed(self.client_id) # TODO: Update to use seed + round + client_id
            pseudo_grad = add_dp_noise_only(pseudo_grad, self.dp_sigma)
            
            # Apply James-Stein estimator if enabled
            if self.jse:
                # Apply James-Stein estimator to each parameter using per-layer dimensionality
                for name, param in pseudo_grad.items():
                    d = param.numel()  # Use dimensionality of this specific parameter tensor
                    pseudo_grad[name] = apply_jse(param, self.dp_sigma, d)
        return res, stats, pseudo_grad

    def _train(self):
        self.model.train()
        
        # Store initial parameters at the start of training
        initial_params = torch.cat([p.detach().clone().flatten() for p in self.model.parameters()])
        
        train_loader, test_loader, client_loaders = get_mnist_loaders(client_num_per_round=10)
        print('DataLoader info:')
        print(f'  Dataset size: {len(train_loader.dataset)}')
        print(f'  Batch size: {train_loader.batch_size}')
        print(f'  Number of batches: {len(train_loader)}')
        
        # Print info about first few mini-batches
        print('\nMini-batch samples:')
        for batch_idx, (x, y) in enumerate(train_loader):
            # if batch_idx = 937:
            print(f'  Batch {batch_idx}:')
            print(f'    Input shape: {x.shape}')
            print(f'    Labels shape: {y.shape}')
            print(f'    Labels: {y.tolist()}')
            if batch_idx >= 2:  # Only show first 3 batches
                break
        
        # Counter to track minibatches
        minibatch_counter = 0
        
        for epoch in range(self.local_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                # print('loss', loss)
                self.optimizer.zero_grad()
                loss.backward()
                
                # Print mini-batch gradients for debugging (first minibatch only)
                if minibatch_counter == 0:
                    print(f"=== FIRST MINI-BATCH GRADIENT (Client {self.client_id}) ===")
                    print(f"Loss value: {loss.item():.6f}")
                    grad_norm = 0
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param_grad_norm = param.grad.norm().item()
                            grad_norm += param_grad_norm ** 2
                            print(f"{name}:")
                            print(f"  Gradient norm: {param_grad_norm:.6f}")
                            print(f"  Gradient mean: {param.grad.mean().item():.6f}")
                            print(f"  Gradient std: {param.grad.std().item():.6f}")
                            print(f"  Gradient shape: {param.grad.shape}")
                            print(f"  First 10 gradient values: {param.grad.flatten()[:10].tolist()}")
                    print(f"Total gradient norm: {grad_norm**0.5:.6f}")
                    print("=" * 60)
                
                self.optimizer.step()
                minibatch_counter += 1
                
                # TODO: the following clipping part is the same as scaffold.py, we should write a function to avoid code duplication
                # Only apply clipping when differential privacy is needed
                if self.dp_sigma > 0:
                    # Calculate delta and apply clipping
                    current_params = torch.cat([p.detach().clone().flatten() for p in self.model.parameters()])
                    delta = current_params - initial_params
                    total_norm = torch.norm(delta)
                    # print('total_norm', total_norm, 'self.clip_bound', self.clip_bound)
                    if total_norm > self.clip_bound:
                        scale = self.clip_bound / total_norm
                        # print('scale', scale)
                        delta = delta * scale
                        current_params = initial_params + delta
                        
                        # Update model parameters
                        start_idx = 0
                        with torch.no_grad():
                            for param in self.model.parameters():
                                num_params = param.numel()
                                param.copy_(current_params[start_idx:start_idx + num_params].view(param.shape))
                                start_idx += num_params
        
        return (
            list(self.model.state_dict(keep_vars=True).values()),
            len(self.trainset.dataset),
        )

    def test(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        loss, correct = self.evaluate()
        stats = {"loss": loss, "correct": correct, "size": len(self.testset)}
        return stats

    def get_client_local_dataset(self):
        datasets = get_dataset(
            self.dataset,
            self.client_id,
        )
        self.trainset = datasets["train"]
        self.valset = datasets["val"]
        self.testset = datasets["test"]

    def _log_while_training(self, evaluate=True, verbose=False, use_valset=False):
        def _log_and_train(*args, **kwargs):
            loss_before = 0
            loss_after = 0
            correct_before = 0
            correct_after = 0
            num_samples = len(self.trainset)
            if evaluate:
                loss_before, correct_before = self.evaluate(use_valset)

            res = self._train(*args, **kwargs)

            if evaluate:
                loss_after, correct_after = self.evaluate(use_valset)

            if verbose:
                
                self.logger.log(
                    "client [{}]   [bold red]loss: {:.4f} -> {:.4f}    [bold blue]accuracy: {:.2f}% -> {:.2f}% (Correct samples: {} -> {})".format(
                        self.client_id,
                        loss_before / num_samples,
                        loss_after / num_samples,
                        correct_before / num_samples * 100.0,
                        correct_after / num_samples * 100.0,
                        correct_before,
                        correct_after,
                    )
                )

            stats = {
                "correct": correct_before,
                "size": num_samples,
            }
            return res, stats

        return _log_and_train

    def set_parameters(self, model_params: OrderedDict):
        self.model.load_state_dict(model_params, strict=False)
        if self.client_id in self.untrainable_params.keys():
            self.model.load_state_dict(
                self.untrainable_params[self.client_id], strict=False
            )

    def get_data_batch(self):
        '''
        Only used in scaffold.py, not in base.py
        '''
        # Set seed for reproducible batch sampling
        np.random.seed(1)
        
        batch_size = (
            self.batch_size
            if self.batch_size > 0
            else int(len(self.trainset) / self.local_epochs)
        )
        indices = torch.from_numpy(
            np.random.choice(self.trainset.indices, batch_size)
        ).long()
        data, targets = self.trainset.dataset[indices]
        return data.to(self.device), targets.to(self.device)
