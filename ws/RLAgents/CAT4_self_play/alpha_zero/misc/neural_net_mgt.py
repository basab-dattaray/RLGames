import os
import sys
import time
from collections import namedtuple

import numpy as np

from ws.RLAgents.CAT4_self_play.alpha_zero._game.othello._ml_lib.pytorch.NeuralNet import NeuralNet
from ws.RLAgents.CAT4_self_play.alpha_zero.misc.average_mgt import average_mgt
from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt

sys.path.append('../../')
from ws.RLUtils.common.DotDict import *


import torch
import torch.optim as optim

# from ws.RLAgents.self_play.alpha_zero._game.othello._ml_lib.pytorch.NeuralNet import NeuralNet

def neural_net_mgt(args):
    nn_args = DotDict({
        'lr': 0.001,
        'dropout': 0.3,
        # 'NUM_EPOCHS': 2,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': 512,
    })

    folder = args.MODEL_PATH_

    def fn_get_untrained_model(arguments):
        # nn_args = nn_args
        untrained_nn = NeuralNet(arguments.game_mgr, nn_args)

        if nn_args.cuda:
            untrained_nn.cuda()
        return untrained_nn

    nnet = fn_get_untrained_model(args)
    board_x, board_y = args.BOARD_SIZE, args.BOARD_SIZE
    action_size = args.game_mgr.fn_get_action_size()

    # @tracer(nn_args)
    def fn_adjust_model_from_examples(examples):
        """
        examples: list of examples, each example is of form (board_pieces, policy, v)
        """
        optimizer = optim.Adam(nnet.parameters())
        fn_count_event, fn_stop_counting = progress_count_mgt('Epochs', args.NUM_EPOCHS)
        for epoch in range(args.NUM_EPOCHS):
            # nn_args.CALL_TRACER_.fn_write(f'Epoch {epoch + 1} of {nn_args.NUM_EPOCHS}')
            fn_count_event()

            nnet.train()
            pi_losses = average_mgt()
            v_losses = average_mgt()

            batch_count = int(len(examples) / nn_args.batch_size)

            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=nn_args.batch_size)
                batch_of_states_as_tuple, batch_of_policies_as_tuple, batch_of_results_as_tuple = list(zip(*[examples[i] for i in sample_ids]))
                batch_of_states = torch.FloatTensor(np.array(batch_of_states_as_tuple).astype(np.float64))
                batch_of_policies = torch.FloatTensor(np.array(batch_of_policies_as_tuple))
                batch_of_results = torch.FloatTensor(np.array(batch_of_results_as_tuple).astype(np.float64))

                # fn_neural_predict
                if nn_args.cuda:
                    batch_of_states, batch_of_policies, batch_of_results = batch_of_states.contiguous().cuda(), batch_of_policies.contiguous().cuda(), batch_of_results.contiguous().cuda()

                # compute output
                batch_of_predicted_policies, batch_of_predicted_results = nnet.forward(batch_of_states)
                loss_policies = _fn_loss_for_policies(batch_of_policies, batch_of_predicted_policies)
                loss_values = _fn_loss_for_values(batch_of_results, batch_of_predicted_results)
                total_loss = loss_policies + loss_values

                # record loss
                pi_losses.fn_update(loss_policies.item(), batch_of_states.size(0))
                v_losses.fn_update(loss_values.item(), batch_of_states.size(0))

                # compute gradient and execute Stochastic Gradient Decent
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        fn_stop_counting()
        # args.CALL_TRACER_.fn_write(f'Number of Epochs for training new model: {args.NUM_EPOCHS}')

    def fn_neural_predict(board):
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if nn_args.cuda: board = board.contiguous().cuda()
        board = board.view(1, board_x, board_y)
        nnet.eval()
        with torch.no_grad():
            policy, value = nnet(board)

        return torch.exp(policy).data.cpu().numpy()[0], value.data.cpu().numpy()[0]

    def _fn_loss_for_policies(actual_policies, predicted_policies):
        loss = -torch.sum(actual_policies * predicted_policies) / actual_policies.size()[0]
        return loss

    def _fn_loss_for_values(actual_results, predicted_results):
        loss = torch.sum((actual_results - predicted_results.view(-1)) ** 2) / actual_results.size()[0]
        return loss

    def fn_save_model(filename= args['MODEL_NAME_']):

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        torch.save({
            'state_dict': nnet.state_dict(),
        }, filepath)

    def fn_load_model(filename= args['MODEL_NAME_']):
        # folder = os.path.join(args.DEMO_FOLDER_PATH_, args.RESULTS_FOLDER_PATH_)
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            return False

        map_location = None if nn_args.cuda else 'cpu'
        model = torch.load(filepath, map_location=map_location)
        nnet.load_state_dict(model['state_dict'])
        return True

    def fn_is_model_available(rel_folder):
        folder = os.path.join(args.DEMO_FOLDER_PATH_, rel_folder)
        filepath = os.path.join(folder, args.MODEL_NAME_)
        if  os.path.exists(filepath):
            return True
        else:
            return False

    neural_net_mgr = namedtuple('_', [
        'fn_get_untrained_model',
        'fn_adjust_model_from_examples',
        'fn_load_model' ,
        'fn_save_model',
        'fn_neural_predict',
        'fn_is_model_available'
    ])

    neural_net_mgr.fn_get_untrained_model = fn_get_untrained_model
    neural_net_mgr.fn_adjust_model_from_examples = fn_adjust_model_from_examples
    neural_net_mgr.fn_load_model = fn_load_model
    neural_net_mgr.fn_save_model = fn_save_model
    neural_net_mgr.fn_neural_predict = fn_neural_predict
    neural_net_mgr.fn_is_model_available = fn_is_model_available

    return neural_net_mgr


