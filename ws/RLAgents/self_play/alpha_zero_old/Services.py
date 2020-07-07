import os
import shutil

from ws.RLAgents.self_play.alpha_zero_old.play.Arena import Arena
from ws.RLAgents.self_play.alpha_zero_old.search.MctsSelector import MctsSelector
from ws.RLAgents.self_play.alpha_zero_old.search.PolicyBroker import PolicyBroker
from ws.RLEnvironments.othello.OthelloGame import OthelloGame as Game, OthelloGame
from ws.RLInterfaces.PARAM_KEY_NAMES import ARCHIVES
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgr
from ws.RLAgents.self_play.alpha_zero_old.Persister import Persister
from ws.RLUtils.monitoring.tracing.prnt_mgt import prnt_mgr


class Services():
    # LOG_REL_PATH = './logs'
    # MODEL_REL_PATH = './models'
    def __init__(self, config_params, calling_filepath):

        self.args = config_params.args
        filepathname_parts = calling_filepath.rsplit('/', 1)
        self.cwd = filepathname_parts[0]
        self.persister = Persister(self.cwd)

        self.fn_print_progress, self.fn_print_end = prnt_mgr(indent=4)

        self.chart = None

        self.archive_path = self.cwd.replace('Demos', ARCHIVES)

        self.game = Game(self.args.board_size)

        self.arena = Arena(self, self.game)

        self.mcts_framework_cls = lambda nn, is_new_mcts: MctsSelector(self.game, nn, self.args, is_new_mcts)

        self.policy_broker = PolicyBroker(self)

        if 'num_testing_eval_games' not in self.args:
            self.args['num_testing_eval_games'] = self.args.num_training_eval_games

        if 'num_iterations' not in self.args:
            self.args['num_iterations'] = 1

        if 'num_model_upgrades' not in self.args:
            self.args['num_model_upgrades'] = self.args['num_iterations']

        self.fn_record = log_mgr(log_dir=self.persister.session_folder, show_debug=False, log_file_name='session.txt')

        pass



    def fn_set_demo_mode(self, demo_name):
        if demo_name == 'pit_against_random':
            pass







