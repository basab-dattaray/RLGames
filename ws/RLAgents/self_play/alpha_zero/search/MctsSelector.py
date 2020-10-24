from ws.RLAgents.self_play.alpha_zero.search.MctsAdaptor import MctsAdapter
from ws.RLAgents.self_play.alpha_zero.search.non_recursive.Mcts import Mcts
from ws.RLAgents.self_play.alpha_zero.search.recursive.MCTS import MCTS


class MctsSelector():

    def __init__(self, game, nnet, args):
        if not args.mcts_recursive:
            self.mcts_adapter = MctsAdapter(game, nnet, args)
            max_num_actions = game.fn_get_action_size()

            mcts = Mcts(
                fn_find_next_state = self.mcts_adapter.fn_find_next_state,
                fn_predict_action_probablities=self.mcts_adapter.fn_predict_action_probablities,
                fn_get_valid_actions=self.mcts_adapter.fn_get_valid_actions,
                fn_terminal_state_status= self.mcts_adapter.fn_terminal_state_status,
                num_mcts_simulations=args.numMCTSSims,
                explore_exploit_ratio=args.cpuct,
                max_num_actions=max_num_actions
            )
            self.fn_get_action_probabilities = lambda state, spread_probabilities: mcts.fn_get_action_probabilities(state, spread_probabilities)
        else:
            mcts = MCTS(game, nnet, args)
            self.fn_get_action_probabilities = lambda state, spread_probabilities: mcts.fn_get_action_probabilities(state, spread_probabilities)
        pass



