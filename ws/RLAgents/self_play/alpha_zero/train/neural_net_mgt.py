import inspect
import logging
import os
import sys
import time
from collections import namedtuple

import numpy as np

from ws.RLUtils.monitoring.tracing.tracer import tracer

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

def neuralNetMgr(args, game):

    # args = args
    nnet = OthelloNNet(game, nnet_params)
    board_x, board_y = game.getBoardSize()
    action_size = game.getActionSize()

    if nnet_params.cuda:
        nnet.cuda()

    @tracer(args)
    def fn_adjust_model_from_examples(examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(nnet.parameters())
        for epoch in range(args.epochs):
            args.recorder.fn_record_message(f'Epoch {epoch + 1} of {args.epochs}')

            nnet.train()
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
                out_pi, out_v = nnet(boards)
                l_pi = loss_pi(target_pis, out_pi)
                l_v = loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                # t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if nnet_params.cuda: board = board.contiguous().cuda()
        board = board.view(1, board_x, board_y)
        nnet.eval()
        with torch.no_grad():
            pi, v = nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(rel_folder='checkpoint', filename='checkpoint.pth.tar'):
        folder = os.path.join(args.demo_folder, rel_folder)
        filepath = os.path.join(folder, filename)
        filepath_abs = os.path.abspath(filepath)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        # else:
        #     print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': nnet.state_dict(),
        }, filepath)

    def load_checkpoint(rel_folder='checkpoint', filename='checkpoint.pth.tar'):
        folder = os.path.join(args.demo_folder, rel_folder)
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if nnet_params.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        nnet.load_state_dict(checkpoint['state_dict'])

    ret_refs = namedtuple('_', ['fn_adjust_model_from_examples','load_checkpoint' ,'save_checkpoint', 'predict'])

    ret_refs.fn_adjust_model_from_examples = fn_adjust_model_from_examples
    ret_refs.load_checkpoint = load_checkpoint
    ret_refs.save_checkpoint = save_checkpoint
    ret_refs.predict = predict

    return ret_refs