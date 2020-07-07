from ws.RLAgents.self_play.alpha_zero.search.mcts_pkg.Mcts import Mcts
from ws.RLAgents.self_play.alpha_zero.search.mcts_recursive_pkg.MctsRecursive import MctsRecursive
from ws.RLEnvironments.othello.MctsAdaptor import MctsAdapter


class MctsSelector():

    def __init__(self, game, nnet, args, is_new_mcts):

        if is_new_mcts:
            self.mcts_adapter = MctsAdapter(game, nnet, args)
            max_num_actions = args.board_size ** 2
            mcts = Mcts(
                fn_find_next_state = self.mcts_adapter.fn_find_next_state,
                fn_predict_action_probablities=self.mcts_adapter.fn_predict_action_probablities,
                fn_get_valid_actions=self.mcts_adapter.fn_get_valid_actions,
                fn_terminal_state_status= self.mcts_adapter.fn_terminal_state_status,
                num_mcts_simulations=args.num_of_mcts_simulations,
                explore_exploit_ratio=args.cpuct,
                max_num_actions=max_num_actions
            )
            self.getActionProb = lambda board: mcts.fn_get_action_probabilities(board)
        else:
            old_mcts = MctsRecursive(game, nnet, args)
            self.getActionProb = lambda canonicalBoard: old_mcts.getActionProb(canonicalBoard)
        pass
