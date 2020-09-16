import inspect
import logging
import os
import sys
import time

import numpy as np


sys.path.append('../../')
from ws.RLAgents.self_play.alpha_zero.misc.utils import *
from ws.RLInterfaces.NeuralNet import NeuralNet

import torch
import torch.optim as optim

from ws.RLAgents.self_play.alpha_zero._game.othello._ml_lib.pytorch.OthelloNNet import OthelloNNet

log = logging.getLogger(__name__)

nnet_params = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    # 'epochs': 2,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

class NeuralNetWrapper(NeuralNet):
    def __init__(self, args, game):
        self.args = args
        self.nnet = OthelloNNet(game, nnet_params)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if nnet_params.cuda:
            self.nnet.cuda()

    def fn_model_from_examples(self, examples):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        self.args.fn_record('    Start Training')
        for epoch in range(self.args.epochs):
            self.args.fn_record(f'     Epoch {epoch + 1} of {self.args.epochs}')
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / nnet_params.batch_size)

            t = range(batch_count)
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=nnet_params.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if nnet_params.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                # t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        self.args.recorder.fn_record_func_title_end()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if nnet_params.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, rel_folder='checkpoint', filename='checkpoint.pth.tar'):
        folder = os.path.join(self.args.demo_folder, rel_folder)
        filepath = os.path.join(folder, filename)
        filepath_abs = os.path.abspath(filepath)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        # else:
        #     print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, rel_folder='checkpoint', filename='checkpoint.pth.tar'):
        folder = os.path.join(self.args.demo_folder, rel_folder)
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if nnet_params.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
