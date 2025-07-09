from torch import nn

ARGS = {
    "mnist": (1, 256, 10),
    "emnist": (1, 256, 62),
    "fmnist": (1, 256, 10),
    "cifar10": (3, 400, 10),
    "cifar100": (3, 400, 100),
}

class SimpleMLP(nn.Module):    
    def __init__(self, dataset, hidden_dim: int = 10):
        super().__init__()
        # For MNIST: 28*28 = 784 input features
        input_dim = 28 * 28 if dataset == "mnist" else 32 * 32 * ARGS[dataset][0]
        num_classes = ARGS[dataset][2]
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x) 


class LeNet5(nn.Module):
    def __init__(self, dataset) -> None:
        super(LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ARGS[dataset][0], 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(ARGS[dataset][1], 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, ARGS[dataset][2]),
        )

    def forward(self, x):
        return self.net(x)
