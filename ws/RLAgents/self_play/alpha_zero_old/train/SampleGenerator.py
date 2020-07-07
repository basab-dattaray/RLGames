from collections import deque

import numpy

from ws.RLInterfaces.PolicyTypes import PolicyTypes

class SampleGenerator():

    def __init__(self, services, nnet):
        self.services = services
        self.mcts = services.mcts_framework_cls(nnet, is_new_mcts=True)

        self.fn_policy = self.services.policy_broker.fn_get_best_action_policy(PolicyTypes.POLICY_TRAINING_EXECUTION, nnet)  # MCTS

    def executeEpisode(self):
        training_samples = []
        board = self.services.game.getInitBoard()
        current_player = 1
        episodeStep = 0
        early_completion_result = None

        while True:
            episodeStep += 1
            canonicalBoard = self.services.game.getCanonicalForm(board, current_player)

            # The canonical board allows either player to use the same model and tree search
            action = self.fn_policy(canonicalBoard)

            valid_moves = self.services.game.getValidMoves(board, current_player)
            if valid_moves[action] == 0:
                score = numpy.sum(board)
                run_result = 0 if score == 0 else (1 if score > 0 else -1)
                current_player_when_terminal_move_happened = current_player * -1 # terminal move happened in the previous turn when the previous player was playing
                return self.fn_form_sample_data(current_player_when_terminal_move_happened, run_result, training_samples, early_termination= True)

            pi = self.mcts.getActionProb(canonicalBoard)
            generated_symmetries = self.services.game.getSymmetries(canonicalBoard, pi)

            for symetry_num in range(len(generated_symmetries)):
                b, p = generated_symmetries[symetry_num]
                training_samples.append([b, current_player, p, None])

            board, current_player, early_completion_result = self.services.game.getNextState(board, current_player, action)

            run_result = self.services.game.getGameEnded(board, current_player) if early_completion_result is None else early_completion_result

            if run_result != 0:
                return self.fn_form_sample_data(current_player, run_result, training_samples)

    def fn_form_sample_data(self, current_player, run_result, training_samples, early_termination = False):
        sample_data = []
        for x in training_samples:
            state, player, action_prob = x[0], x[1], x[2]
            result = run_result * (1 if (player == current_player) else -1)
            sample_data.append([state, action_prob, result])
        return sample_data

    def fn_generate_samples(self):
        all_samples = deque([], maxlen=self.services.args.sample_buffer_size)

        # self.services.fn_record(f'Generates Samples with {self.args.num_episodes} Episodes')
        for episode_num in range(1, self.services.args.num_episodes + 1):
            samples = self.executeEpisode()
            if samples is not None:
                all_samples += samples

                self.services.fn_record(f'  SAMPLE GEN EPISODE {episode_num} of {self.services.args.num_episodes}:  {len(samples)} samples generated')
            else:
                self.services.fn_record(f'  SAMPLE GEN EPISODE {episode_num} of {self.services.args.num_episodes}:  !!! No Samples were generated')
        self.services.fn_record(f'Total Samples Collected: {len(all_samples)}')
        return all_samples
