import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, game_mgr, nn_args):
        # game params
        # self.board_x, self.board_y = game_mgr.fn_get_board_size(), game_mgr.fn_get_board_size()
        # self.action_size = game_mgr.fn_get_action_size()
        self.app_info = nn_args

        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, nn_args.NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nn_args.NUM_CHANNELS, nn_args.NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nn_args.NUM_CHANNELS, nn_args.NUM_CHANNELS, 3, stride=1)
        self.conv4 = nn.Conv2d(nn_args.NUM_CHANNELS, nn_args.NUM_CHANNELS, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(nn_args.NUM_CHANNELS)
        self.bn2 = nn.BatchNorm2d(nn_args.NUM_CHANNELS)
        self.bn3 = nn.BatchNorm2d(nn_args.NUM_CHANNELS)
        self.bn4 = nn.BatchNorm2d(nn_args.NUM_CHANNELS)

        self.fc1 = nn.Linear(nn_args.NUM_CHANNELS * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           state: BATCH_SIZE x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # BATCH_SIZE x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # BATCH_SIZE x NUM_CHANNELS x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # BATCH_SIZE x NUM_CHANNELS x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # BATCH_SIZE x NUM_CHANNELS x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # BATCH_SIZE x NUM_CHANNELS x (board_x-4) x (board_y-4)
        s = s.view(-1, self.app_info.NUM_CHANNELS*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.app_info.DROPOUT, training=self.training)  # BATCH_SIZE x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.app_info.DROPOUT, training=self.training)  # BATCH_SIZE x 512

        pi = self.fc3(s)                                                                         # BATCH_SIZE x _action_size
        v = self.fc4(s)                                                                          # BATCH_SIZE x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
