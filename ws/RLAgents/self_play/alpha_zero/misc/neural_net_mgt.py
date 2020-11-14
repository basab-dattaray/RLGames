import os
import sys
import time
from collections import namedtuple

import numpy as np

from ws.RLAgents.self_play.alpha_zero.misc.AverageMeter import AverageMeter
from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt

sys.path.append('../../')
from ws.RLUtils.common.DotDict import *


import torch
import torch.optim as optim

from ws.RLAgents.self_play.alpha_zero._game.othello._ml_lib.pytorch.NeuralNet import NeuralNet




def neural_net_mgt(args):
    nn_args = DotDict({
        'lr': 0.001,
        'dropout': 0.3,
        # 'epochs': 2,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': 512,
    })

    def fn_get_untrained_model(arguments):
        # nn_args = nn_args
        untrained_nn = NeuralNet(arguments.game_mgr, nn_args)

        if nn_args.cuda:
            untrained_nn.cuda()
        return untrained_nn

    nnet = fn_get_untrained_model(args)
    board_x, board_y = args.board_size, args.board_size
    action_size = args.game_mgr.fn_get_action_size()

    # @tracer(nn_args)
    def fn_adjust_model_from_examples(examples):
        """
        examples: list of examples, each example is of form (board_pieces, action_probs, v)
        """
        optimizer = optim.Adam(nnet.parameters())
        fn_count_event, fn_stop_counting = progress_count_mgt('Epochs', args.epochs)
        for epoch in range(args.epochs):
            # nn_args.calltracer.fn_write(f'Epoch {epoch + 1} of {nn_args.epochs}')
            fn_count_event()

            nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / nn_args.batch_size)

            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=nn_args.batch_size)
                batch_of_states, batch_of_action_probablities, batch_of_results = list(zip(*[examples[i] for i in sample_ids]))
                batch_of_states = torch.FloatTensor(np.array(batch_of_states).astype(np.float64))
                batch_of_action_probablities_as_nparray = torch.FloatTensor(np.array(batch_of_action_probablities))
                batch_of_results_as_array = torch.FloatTensor(np.array(batch_of_results).astype(np.float64))

                # predict
                if nn_args.cuda:
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
        fn_stop_counting()
        args.calltracer.fn_write(f'Number of Epochs for training new model: {args.epochs}')


    def predict(board):
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if nn_args.cuda: board = board.contiguous().cuda()
        board = board.view(1, board_x, board_y)
        nnet.eval()
        with torch.no_grad():
            pi, v = nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def fn_loss_for_action_probs(targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def fn_loss_for_values(targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def fn_save_model(filename= args['model_name']):
        folder = os.path.join(args.demo_folder, args.rel_model_path)
        filepath = os.path.join(folder, filename)
        filepath_abs = os.path.abspath(filepath)
        if not os.path.exists(folder):
            os.mkdir(folder)

        torch.save({
            'state_dict': nnet.state_dict(),
        }, filepath)

    def fn_load_model(filename= args['model_name']):
        folder = os.path.join(args.demo_folder, args.rel_model_path)
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            return False

        map_location = None if nn_args.cuda else 'cpu'
        model = torch.load(filepath, map_location=map_location)
        nnet.load_state_dict(model['state_dict'])
        return True

    def fn_is_model_available(rel_folder):
        folder = os.path.join(args.demo_folder, rel_folder)
        filepath = os.path.join(folder, args.model_name)
        if  os.path.exists(filepath):
            return True
        else:
            return False


    neural_net_mgr = namedtuple('_', [
        'fn_get_untrained_model',
        'fn_adjust_model_from_examples',
        'fn_load_model' ,
        'fn_save_model',
        'predict',
        'fn_is_model_available'
    ])

    neural_net_mgr.fn_get_untrained_model = fn_get_untrained_model
    neural_net_mgr.fn_adjust_model_from_examples = fn_adjust_model_from_examples
    neural_net_mgr.fn_load_model = fn_load_model
    neural_net_mgr.fn_save_model = fn_save_model
    neural_net_mgr.predict = predict
    neural_net_mgr.fn_is_model_available = fn_is_model_available

    return neural_net_mgr


