import os
import sys
import time

import numpy as np
# from tqdm import tqdm

# from ws.RLUtils.monitoring.charting.Chart import Chart

sys.path.append('../../')
from ws.RLAgents.self_play.alpha_zero_old.othello.pytorch.utils import *
from ws.RLInterfaces.NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .OthelloNNet import OthelloNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, services):
        if 'epochs' in services.args.keys():
            args.epochs = services.args['epochs']

        self.services = services
        self.board_x = self.board_y = services.args.board_size
        self.action_size = services.args.board_size ** 2

        self.nnet = onnet(services.args.board_size, args)


        if args.cuda:
            self.nnet.cuda()

    def train(self, examples, iteration_info, chart):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        self.chart = chart
        optimizer = optim.Adam(self.nnet.parameters())

        self.services.fn_record(f'TRAINING runs in {args.epochs} epochs')
        for epoch in range(1, args.epochs + 1):
            epoch_info = {'max_epochs': args.epochs, 'epoch_index': epoch}
            self.services.fn_record(f'  TRAINING EPOCH {epoch} out of {args.epochs}')

            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            # t = tqdm(range(batch_count), desc='Training Net')
            for batch_index in range(batch_count):
                batch_info = {'max_batches': batch_count, 'batch_index': batch_index + 1}
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)

                model_loss_val_fraction = 0.5
                if 'model_loss_value_fraction' in self.services.args:
                    if self.services.args.model_loss_val_fraction > 0.0 and self.services.args.model_loss_val_fraction < 1:
                        model_loss_val_fraction = self.services.args.model_loss_val_fraction

                total_loss = (1 - model_loss_val_fraction) * l_pi + model_loss_val_fraction * l_v

                self.chart.fn_record_event(None,
                                           [total_loss.item(), l_v.item(), l_pi.item()]
                                           )

                progress_info = {**iteration_info, **epoch_info, **batch_info}
                self.chart.fn_update_title(progress_info)

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                # t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()


    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def fn_predict_action(self, board):

        action_probabilities = self.predict(board)[0]

        valid_moves =self.services.game.getValidMoves(board, 1)

        valid_actions = action_probabilities * valid_moves
        adjusted_valid_actions = valid_actions

        return adjusted_valid_actions

    def fn_get_best_predicted_action (self, board):
        action_probabilities = self.predict(board)[0]

        best_action = np.argmax(action_probabilities)

        return best_action

    def fn_get_best_action_policy_func(self, fn_predict_action, is_stochastic = True):

        def fn_best_stochastic_action(state):
            valid_actions = fn_predict_action(state)
            total = sum(valid_actions)
            if total == 0:
                best_action = np.argmax(valid_actions)
                return best_action

            action_probabilities = valid_actions/total
            best_action = np.random.choice(len(action_probabilities), p=action_probabilities)
            return best_action

        if is_stochastic:
            return fn_best_stochastic_action
        else:
            return lambda state: np.argmax(fn_predict_action(state))



    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_the_model(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder, filename):

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))

        map_location = None if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
