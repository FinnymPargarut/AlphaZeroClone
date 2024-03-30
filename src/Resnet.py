import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.game = game

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(13, num_hidden, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * game.row_count * game.column_count, game.action_size * 2 + game.promotion_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 6, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        start_policy, end_policy = policy[:, :self.game.action_size], policy[:, self.game.action_size:]
        value = self.valueHead(x)
        return start_policy, end_policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
