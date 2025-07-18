import os
import pickle
import random
from argparse import Namespace
from collections import OrderedDict

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm
from torch.utils.data import DataLoader

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from config.models import LeNet5, SimpleMLP
from config.util import (
    DATA_DIR,
    LOG_DIR,
    PROJECT_DIR,
    TEMP_DIR,
    clone_parameters,
    fix_random_seed,
)

sys.path.append(PROJECT_DIR)
sys.path.append(DATA_DIR)
from client.base import ClientBase
from data.utils.util import get_client_id_indices

fix_random_seed(1)

class ServerBase:
    def __init__(self, args: Namespace, algo: str):
        self.algo = algo
        self.args = args
        # default log file format
        self.log_name = "{}_{}_{}_{}.html".format(
            self.algo,
            self.args.dataset,
            self.args.global_epochs,
            self.args.local_epochs,
        )
        self.device = torch.device(
            "cuda" if self.args.gpu and torch.cuda.is_available() else "cpu"
        )
        
        # fix_random_seed(self.args.seed)
        fix_random_seed(1)

        self.backbone = SimpleMLP  # LeNet5
        self.logger = Console(
            record=True,
            log_path=False,
            log_time=False,
        )
        self.client_id_indices, self.client_num_in_total = get_client_id_indices(
            self.args.dataset
        )
        self.temp_dir = TEMP_DIR / self.algo
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        _dummy_model = self.backbone(self.args.dataset).to(self.device)
        passed_epoch = 0
        self.global_params_dict: OrderedDict[str : torch.Tensor] = None
     
        self.global_params_dict = OrderedDict(
            _dummy_model.state_dict(keep_vars=True)
        )

        self.global_epochs = self.args.global_epochs - passed_epoch
        self.logger.log("Backbone:", _dummy_model)

        self.trainer: ClientBase = None
        self.num_correct = [[] for _ in range(self.global_epochs)]
        self.num_samples = [[] for _ in range(self.global_epochs)]
        # Track global model performance
        self.global_train_loss = []
        self.global_train_acc = []
        
        # Load the full training and test datasets
        pickles_dir = Path(PROJECT_DIR) / "data" / self.args.dataset / "pickles"
        with open(pickles_dir / "full_trainset.pkl", "rb") as f:
            self.full_trainset = pickle.load(f)
        with open(pickles_dir / "full_testset.pkl", "rb") as f:
            self.full_testset = pickle.load(f)

    @torch.no_grad()
    def evaluate_global_model(self):
        """Evaluate the global model on the complete training dataset."""
        if self.full_trainset is None:
            raise RuntimeError("Full training dataset has not been loaded.")
            
        # Create a model with global parameters
        model = self.backbone(self.args.dataset).to(self.device)
        model.load_state_dict(self.global_params_dict)
        model.eval()
        
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        dataloader = DataLoader(self.full_trainset, batch_size=128)
        
        total_loss = 0
        total_correct = 0
        total_samples = len(self.full_trainset)
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            pred = torch.softmax(logits, -1).argmax(-1)
            total_correct += (pred == y).int().sum().item()
        print(">>>>> total number of samples: ", total_samples, " >>>> total correct: ", total_correct)
        
        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100.0
        
        return avg_loss, accuracy

    def train(self):
        self.logger.log("=" * 30, "TRAINING", "=" * 30, style="bold green")
        progress_bar = (
            track(
                range(self.global_epochs),
                "[bold green]Training...",
                console=self.logger,
            )
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training...")
        )
        for E in progress_bar:
            if E % self.args.verbose_gap == 0:
                self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )
            pseudo_grad_cache = []
            res_cache = []
            for client_id in selected_clients:
                client_local_params = clone_parameters(self.global_params_dict)
                res, stats, pseudo_grad = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    epoch=E,
                    verbose=(E % self.args.verbose_gap) == 0,
                )
                res_cache.append(res)
                pseudo_grad_cache.append((pseudo_grad, res[1]))
                self.num_correct[E].append(stats["correct"])
                self.num_samples[E].append(stats["size"])
            self.aggregate(pseudo_grad_cache, res_cache)

            # Evaluate global model on all training data after aggregation
            global_loss, global_acc = self.evaluate_global_model()
            self.global_train_loss.append(global_loss)
            self.global_train_acc.append(global_acc)
            
            if E % self.args.verbose_gap == 0:
                self.logger.log(
                    "[bold green]Global Model Performance on All Training Data:[/bold green]"
                    f" Loss: {global_loss:.4f}, Accuracy: {global_acc:.2f}%"
                )

            if E % self.args.save_period == 0:
                torch.save(
                    self.global_params_dict,
                    self.temp_dir / "global_model.pt",
                )
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
                # Save global model performance metrics
                with open(self.temp_dir / "global_metrics.pkl", "wb") as f:
                    pickle.dump({
                        "loss": self.global_train_loss,
                        "accuracy": self.global_train_acc
                    }, f)

    @torch.no_grad()
    def aggregate(self, pseudo_grad_cache,res_cache):
        pseudo_grads, weights_cache = list(zip(*pseudo_grad_cache))
        weight_sum = sum(weights_cache)
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum

        for key in self.global_params_dict.keys():
            aggregated_pseudo_grad_for_key = torch.sum(
                torch.stack([grad[key] for grad in pseudo_grads], dim=-1) * weights,
                dim=-1,
            )
            self.global_params_dict[key] += aggregated_pseudo_grad_for_key

    def test(self) -> None:
        self.logger.log("=" * 30, "TESTING", "=" * 30, style="bold blue")
        
        if self.full_testset is None:
            raise RuntimeError("Full test dataset has not been loaded.")
            
        # Create a model with global parameters
        model = self.backbone(self.args.dataset).to(self.device)
        model.load_state_dict(self.global_params_dict)
        model.eval()
        
        # Create dataloader for the complete test dataset
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        dataloader = DataLoader(self.full_testset, batch_size=128, shuffle=False)
        
        total_loss = 0
        total_correct = 0
        total_samples = len(self.full_testset)
        
        with torch.no_grad():
            for x, y in track(
                dataloader,
                "[bold blue]Testing...",
                console=self.logger,
                disable=self.args.log,
            ):
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                total_loss += criterion(logits, y).item()
                pred = torch.softmax(logits, -1).argmax(-1)
                total_correct += (pred == y).int().sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100.0
        
        self.logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
        self.logger.log(
            "Global Model Performance on Complete Test Dataset:"
        )
        self.logger.log(
            "loss: {:.4f}    accuracy: {:.2f}%".format(
                avg_loss,
                accuracy,
            )
        )
        
        # Track when certain accuracy thresholds are reached
        acc_range = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
        min_acc_idx = 10
        max_acc = 0
        for E, (corr, n) in enumerate(zip(self.num_correct, self.num_samples)):
            avg_acc = sum(corr) / sum(n) * 100.0
            for i, acc in enumerate(acc_range):
                if avg_acc >= acc and avg_acc > max_acc:
                    self.logger.log(
                        "{} achieved {}% accuracy({:.2f}%) at epoch: {}".format(
                            self.algo, acc, avg_acc, E
                        )
                    )
                    max_acc = avg_acc
                    min_acc_idx = i
                    break
            acc_range = acc_range[:min_acc_idx]

    def run(self):
        self.logger.log("Arguments:", dict(self.args._get_kwargs()))
        self.train()
        self.test()
        if self.args.log:
            if not os.path.isdir(LOG_DIR):
                os.mkdir(LOG_DIR)
            self.logger.save_html(LOG_DIR / self.log_name)

        # delete all temporary files
        if os.listdir(self.temp_dir) != []:
            os.system(f"rm -rf {self.temp_dir}")
