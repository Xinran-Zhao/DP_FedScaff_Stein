from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict

import torch
from rich.console import Console

from .base import ClientBase
from src.config.util import add_dp_noise, add_dp_noise_only


class SCAFFOLDClient(ClientBase):
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
    ):
        super(SCAFFOLDClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
            dp_sigma,
            clip_bound,
        )
        self.c_local: Dict[List[torch.Tensor]] = {}
        self.c_diff = []

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        c_global,
        epoch: int,
        evaluate=True,
        verbose=True,
        use_valset=True,
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        if self.client_id not in self.c_local.keys():
            self.c_diff = c_global
        else:
            self.c_diff = []
            for c_l, c_g in zip(self.c_local[self.client_id], c_global):
                self.c_diff.append(-c_l + c_g)
        _, stats = self._log_while_training(evaluate, verbose, use_valset)()
        # update local control variate
        with torch.no_grad():
            trainable_parameters = filter(
                lambda p: p.requires_grad, model_params.values()
            )

            if self.client_id not in self.c_local.keys():
                self.c_local[self.client_id] = [
                    torch.zeros_like(param, device=self.device)
                    for param in self.model.parameters()
                ]

            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)
            y_delta_dict = OrderedDict()
            trainable_param_names = [name for name, param in model_params.items() if param.requires_grad]
            
            for i, (param_l, param_g) in enumerate(zip(self.model.parameters(), trainable_parameters)):
                pseudo_gradient = param_l - param_g
                if i < len(trainable_param_names):
                    y_delta_dict[trainable_param_names[i]] = pseudo_gradient
            
            # Apply DP noise only (clipping already done during local training in SCAFFOLD)
            if self.dp_sigma > 0:
                seed = epoch * 1000 + self.client_id
                y_delta_dict = add_dp_noise_only(y_delta_dict, self.dp_sigma, seed)
            
            # Convert back to list format
            y_delta = [y_delta_dict[name] for name in trainable_param_names]

            # compute c_plus
            coef = 1 / (self.local_epochs * self.local_lr)
            for c_l, c_g, diff in zip(self.c_local[self.client_id], c_global, y_delta):
                c_plus.append(c_l - c_g - coef * diff)

            # compute c_delta
            for c_p, c_l in zip(c_plus, self.c_local[self.client_id]):
                c_delta.append(c_p - c_l)

            self.c_local[self.client_id] = c_plus

        if self.client_id not in self.untrainable_params.keys():
            self.untrainable_params[self.client_id] = {}
        for name, param in self.model.state_dict(keep_vars=True).items():
            if not param.requires_grad:
                self.untrainable_params[self.client_id][name] = param.clone()

        return (y_delta, c_delta), stats

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
            for param, c_d in zip(self.model.parameters(), self.c_diff):
                param.grad += c_d.data
            self.optimizer.step()
            
            # Apply clipping when differential privacy is needed (same as base class)
            if self.dp_sigma > 0:
                # Calculate delta and apply clipping
                current_params = torch.cat([p.detach().clone().flatten() for p in self.model.parameters()])
                delta = current_params - initial_params
                total_norm = torch.norm(delta)
                
                if total_norm > self.clip_bound:
                    scale = self.clip_bound / total_norm
                    delta = delta * scale
                    current_params = initial_params + delta
                    
                    # Update model parameters with clipped delta
                    start_idx = 0
                    with torch.no_grad():
                        for param in self.model.parameters():
                            num_params = param.numel()
                            param.copy_(current_params[start_idx:start_idx + num_params].view(param.shape))
                            start_idx += num_params
