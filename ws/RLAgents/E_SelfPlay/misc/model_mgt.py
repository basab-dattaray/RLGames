import os
import time
from collections import namedtuple

import numpy as np

from ws.RLAgents.E_SelfPlay._game.othello._ml_lib.pytorch.NeuralNet import NeuralNet
from ws.RLAgents.E_SelfPlay.misc.average_mgt import average_mgt
from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt

# sys.dir_path.append('../../')
from ws.RLUtils.common.DotDict import *


import torch
import torch.optim as optim

def model_mgt(game_mgr, model_folder_path):
    nn_args = DotDict({
        'BATCH_SIZE': 64,
        'IS_CUDA': torch.cuda.is_available(),
        'NUM_CHANNELS': 512,
        'DROPOUT': 0.3,
    })

    _MODEL_NAME = 'model.tar'

    def fn_save_model(model_file_name= None):
        if model_file_name is None:
            model_file_name = _MODEL_NAME
        filepath = os.path.join(model_folder_path, model_file_name)
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)

        torch.save({
            'state_dict': nnet.state_dict(),
        }, filepath)

    def fn_load_model(model_file_name= None):
        if model_file_name is None:
            model_file_name = _MODEL_NAME

        filepath = os.path.join(model_folder_path, model_file_name)

        if not os.path.exists(filepath):
            return False

        map_location = None if nn_args.IS_CUDA else 'cpu'
        model = torch.load(filepath, map_location=map_location)
        nnet.load_state_dict(model['state_dict'])
        return True

    def fn_is_model_available(results_path):
        filepath = os.path.join(results_path, _MODEL_NAME)
        if  os.path.exists(filepath):
            return True
        else:
            return False

    def _fn_get_untrained_model():
        board_width = board_height = game_mgr.fn_get_board_size()
        action_size = game_mgr.fn_get_action_size()
        untrained_nn = NeuralNet(action_size, (board_width, board_height), nn_args)

        if nn_args.IS_CUDA:
            untrained_nn.cuda()
        return untrained_nn

    nnet = _fn_get_untrained_model()

    # @tracer(nn_args)
    def fn_adjust_model_from_examples(examples, num_epochs):
        optimizer = optim.Adam(nnet.parameters())
        fn_count_event, fn_stop_counting = progress_count_mgt('Epochs', num_epochs)
        for epoch in range(num_epochs):
            fn_count_event()

            nnet.train()
            pi_losses = average_mgt()
            v_losses = average_mgt()

            batch_count = int(len(examples) / nn_args.BATCH_SIZE)

            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=nn_args.BATCH_SIZE)
                batch_of_states_as_tuple, batch_of_policies_as_tuple, batch_of_results_as_tuple = list(zip(*[examples[i] for i in sample_ids]))
                batch_of_states = torch.FloatTensor(np.array(batch_of_states_as_tuple).astype(np.float64))
                batch_of_policies = torch.FloatTensor(np.array(batch_of_policies_as_tuple))
                batch_of_results = torch.FloatTensor(np.array(batch_of_results_as_tuple).astype(np.float64))

                # fn_neural_predict
                if nn_args.IS_CUDA:
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

    def fn_neural_predict(state):
        start = time.time()

        # preparing input
        state = torch.FloatTensor(state.astype(np.float64))
        if nn_args.IS_CUDA:
            state = state.contiguous().cuda()
        # state = state.view(1, board_x, board_y)
        nnet.eval()
        with torch.no_grad():
            policy, value = nnet(state)

        return torch.exp(policy).data.cpu().numpy()[0], value.data.cpu().numpy()[0]

    def _fn_loss_for_policies(actual_policies, predicted_policies):
        loss = -torch.sum(actual_policies * predicted_policies) / actual_policies.size()[0]
        return loss

    def _fn_loss_for_values(actual_results, predicted_results):
        loss = torch.sum((actual_results - predicted_results.view(-1)) ** 2) / actual_results.size()[0]
        return loss


    neural_net_mgr = namedtuple('_', [
        'fn_adjust_model_from_examples',
        'fn_load_model' ,
        'fn_save_model',
        'fn_neural_predict',
        'fn_is_model_available'
    ])

    neural_net_mgr.fn_adjust_model_from_examples = fn_adjust_model_from_examples
    neural_net_mgr.fn_load_model = fn_load_model
    neural_net_mgr.fn_save_model = fn_save_model
    neural_net_mgr.fn_neural_predict = fn_neural_predict
    neural_net_mgr.fn_is_model_available = fn_is_model_available

    return neural_net_mgr


