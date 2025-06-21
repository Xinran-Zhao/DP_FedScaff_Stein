from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import torch
import numpy as np
from path import Path
from rich.console import Console
from torch.utils.data import Subset, DataLoader

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from data.utils.util import get_dataset, set_seed
from src.config.util import add_dp_noise, add_dp_noise_only, apply_jse


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
        set_seed(0)
        self.client_id: int = None
        self.valset: Subset = None
        self.trainset: Subset = None
        self.testset: Subset = None
        self.model: torch.nn.Module = deepcopy(backbone).to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=local_lr
        )
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
    def evaluate(self, use_valset=True):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = 0
        correct = 0
        dataloader = DataLoader(self.valset if use_valset else self.testset, 32)
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
        use_valset=True,
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
            seed = epoch * 1000 + self.client_id
            pseudo_grad = add_dp_noise_only(pseudo_grad, self.dp_sigma, seed)
            
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
        
        for _ in range(self.local_epochs):
            x, y = self.get_data_batch()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
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

    def _log_while_training(self, evaluate=True, verbose=False, use_valset=True):
        def _log_and_train(*args, **kwargs):
            loss_before = 0
            loss_after = 0
            correct_before = 0
            correct_after = 0
            num_samples = len(self.valset)
            if evaluate:
                loss_before, correct_before = self.evaluate(use_valset)

            res = self._train(*args, **kwargs)

            if evaluate:
                loss_after, correct_after = self.evaluate(use_valset)

            if verbose:
                self.logger.log(
                    "client [{}]   [bold red]loss: {:.4f} -> {:.4f}    [bold blue]accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        loss_before / num_samples,
                        loss_after / num_samples,
                        correct_before / num_samples * 100.0,
                        correct_after / num_samples * 100.0,
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
