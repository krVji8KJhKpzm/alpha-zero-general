import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OthelloNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        if s.dim() == 3:
            B, H, W = s.shape
            S = None
            s_merged = s
        elif s.dim() == 4:
            B, S, H, W = s.shape
            s_merged = s.reshape(B*S, H, W)
        else:
            raise ValueError(f"Expect (B,H,W) or (B,S,H,W), got {tuple(s.shape)}")

        x = s_merged.view(-1, 1, self.board_x, self.board_y)                 # (B[*S], 1, H, W)
        x = F.relu(self.bn1(self.conv1(x)))                                  # (B[*S], C, H, W)
        x = F.relu(self.bn2(self.conv2(x)))                                  # (B[*S], C, H, W)
        x = F.relu(self.bn3(self.conv3(x)))                                  # (B[*S], C, H-2, W-2)
        x = F.relu(self.bn4(self.conv4(x)))                                  # (B[*S], C, H-4, W-4)

        z = x.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))  # (B[*S], hidden)

        h = F.dropout(F.relu(self.fc_bn1(self.fc1(z))), p=self.args.dropout, training=self.training)
        h = F.dropout(F.relu(self.fc_bn2(self.fc2(h))), p=self.args.dropout, training=self.training)

        pi = self.fc3(h)                     # (B[*S], action_size)
        v  = self.fc4(h)                     # (B[*S], 1)

        pi = F.log_softmax(pi, dim=1)
        v  = torch.tanh(v)

        if s.dim() == 4:
            pi = pi.view(B, S, -1)           # (B, 8, action_size) -> 8×65
            v  = v.view(B, S, 1)             # (B, 8, 1)
            z  = z.view(B, S, -1)            # (B, 8, hidden)

            return pi[:, -1, :], pi, v[:, -1, :], v, z
        else:
            return pi, None, v, None, None