import inspect
import logging
import os
import sys
import time
from collections import namedtuple

import numpy as np

from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt
from ws.RLUtils.monitoring.tracing.tracer import tracer

sys.path.append('../../')
from ws.RLAgents.self_play.alpha_zero.misc.utils import *
from ws.RLInterfaces.NeuralNet import NeuralNet

import torch
import torch.optim as optim

from ws.RLAgents.self_play.alpha_zero._game.othello._ml_lib.pytorch.NeuralNet import NeuralNet

log = logging.getLogger(__name__)

nnet_params = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    # 'epochs': 2,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

def neural_net_mgt(args, game):

    # args = args
    nnet = NeuralNet(game, nnet_params)
    board_x, board_y = game.fn_get_board_size(), game.fn_get_board_size()
    action_size = game.fn_get_action_size()

    if nnet_params.cuda:
        nnet.cuda()

    @tracer(args)
    def fn_adjust_model_from_examples(examples):
        """
        examples: list of examples, each example is of form (board_pieces, action_probs, v)
        """
        optimizer = optim.Adam(nnet.parameters())
        fn_count_episode, fn_end_couunting = progress_count_mgt('Epochs', args.epochs)
        for epoch in range(args.epochs):
            # args.recorder.fn_record_message(f'Epoch {epoch + 1} of {args.epochs}')
            fn_count_episode()

            nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / nnet_params.batch_size)

            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=nnet_params.batch_size)
                batch_of_states, batch_of_action_probablities, batch_of_results = list(zip(*[examples[i] for i in sample_ids]))
                batch_of_states = torch.FloatTensor(np.array(batch_of_states).astype(np.float64))
                batch_of_action_probablities_as_nparray = torch.FloatTensor(np.array(batch_of_action_probablities))
                batch_of_results_as_array = torch.FloatTensor(np.array(batch_of_results).astype(np.float64))

                # predict
                if nnet_params.cuda:
                    batch_of_states, batch_of_action_probablities_as_nparray, batch_of_results_as_array = batch_of_states.contiguous().cuda(), batch_of_action_probablities_as_nparray.contiguous().cuda(), batch_of_results_as_array.contiguous().cuda()

                # compute output
                batch_of_predicted_action_probs, batch_of_predicted_values = nnet(batch_of_states)
                loss_action_probablities = fn_loss_for_action_probs(batch_of_action_probablities_as_nparray, batch_of_predicted_action_probs)
                loss_values = fn_loss_for_values(batch_of_results_as_array, batch_of_predicted_values)
                total_loss = loss_action_probablities + loss_values

                # record loss
                pi_losses.update(loss_action_probablities.item(), batch_of_states.size(0))
                v_losses.update(loss_values.item(), batch_of_states.size(0))
                # t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        fn_end_couunting()
        args.recorder.fn_record_message(f'Epochs: {args.epochs}')

    def predict(board):
        """
        board_pieces: np array with board_pieces
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

    def fn_loss_for_action_probs(targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def fn_loss_for_values(targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(rel_folder, filename):
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

    def load_checkpoint(rel_folder, filename):
        folder = os.path.join(args.demo_folder, rel_folder)
        filepath = os.path.join(folder, filename)
        if filename != 'temp.tar':
            if not os.path.exists(filepath):
                raise ("No model in path {}".format(filepath))
        map_location = None if nnet_params.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        nnet.load_state_dict(checkpoint['state_dict'])

    def fn_is_model_available(rel_folder):
        folder = os.path.join(args.demo_folder, rel_folder)
        filepath = os.path.join(folder, 'model.tar')
        if  os.path.exists(filepath):
            return True
        else:
            return False


    ret_refs = namedtuple('_', ['fn_adjust_model_from_examples','load_checkpoint' ,'save_checkpoint', 'predict', 'fn_is_model_available'])

    ret_refs.fn_adjust_model_from_examples = fn_adjust_model_from_examples
    ret_refs.load_checkpoint = load_checkpoint
    ret_refs.save_checkpoint = save_checkpoint
    ret_refs.predict = predict
    ret_refs.fn_is_model_available = fn_is_model_available

    return ret_refs