
import shutil
import signal
import os
from datetime import datetime as dt

from pathlib import Path
from time import time

from ws.RLAgents.self_play.alpha_zero_old.play.Arena import Arena

from ws.RLAgents.self_play.alpha_zero_old.train.Coach import Coach

from ws.RLEnvironments.othello.OthelloGame import OthelloGame as Game, OthelloGame
from ws.RLAgents.self_play.alpha_zero_old.othello.OthelloPlayers import fn_get_random_policy, fn_get_human_policy

from ws.RLAgents.self_play.alpha_zero_old.othello.pytorch.NNetWrapper_old import NNetWrapper_old as NNet
from ws.RLAgents.self_play.alpha_zero_old.Services import Services
from ws.RLInterfaces.PolicyTypes import PolicyTypes
from ws.RLUtils.decorators.breadcrumbs import encapsulate


class Agent():


    def __init__(self, config_params, calling_filepath):
        self.services = Services(config_params, calling_filepath)
        self.config_params = config_params

        self.start_time = time()
        self.demo_name = Path(calling_filepath).stem
        self.services.fn_record(f'@@@ DEMO NAME:: {self.demo_name}')
        self.services.fn_record(f'@@@ PATH:: {calling_filepath}')

    def exit_gracefully(self, signum, frame):

        if self.services.chart is not None:
            self.services.chart.fn_close()
            self.services.fn_record('@@@ Chart Saved')

        self.fn_archive_it()

        self.services.fn_record('TERMINATED EARLY AFTER SAVING MODEL WEIGHTS')
        self.services.fn_record(f'Total Time Taken = {time() - self.start_time} seconds')
        exit()

    @encapsulate
    def fn_train(self):
        self.services.fn_set_demo_mode('train')

        signal.signal(signal.SIGINT, self.exit_gracefully)

        nnet = NNet(self.services)

        c = Coach(self.services, nnet)

        c.learn()

        self.services.fn_record(f'Total Time Taken = {time() - self.start_time} seconds')

        return self

    def fn_test_policy(self, policy_to_test, game, verbose=False, num_of_games= None):

        signal.signal(signal.SIGINT, self.exit_gracefully)

        if num_of_games is None:
            num_of_games = self.services.args.num_testing_eval_games

        self.services.fn_set_demo_mode('pit_against_random')

        nnet = NNet(self.services)

        msg = self.services.persister.fn_load_model(nnet)
        if msg is not None:
            self.services.fn_record(msg)
            exit()
        else:
            self.services.fn_record("@@@ Model File Loaded")

        system_policy = self.services.policy_broker.fn_get_best_action_policy(PolicyTypes.POLICY_TESTING_EVAL, nnet)

        arena = Arena(self.services, game, display=OthelloGame.display)

        system_policy_wins, test_policy_wins, draws = arena.playGames(
            system_policy,
            policy_to_test,
            num_of_games,
            verbose= verbose)

        total_runs = system_policy_wins + test_policy_wins + draws
        effective_system_policy_wins = system_policy_wins + draws/2
        system_policy_win_trend = effective_system_policy_wins/total_runs * 100
        self.services.fn_record(f'  SYSTEM_POLICY WINS versus RANDOM POLICY WINS : {system_policy_wins} / {test_policy_wins} ; DRAWS : {draws};     Sys Win Trending: {system_policy_win_trend}%')


        pass

    @encapsulate
    def fn_pit_against_random(self):
        game = Game(self.services.args.board_size)
        fn_random_policy = fn_get_random_policy(game).fn_play_it
        self.fn_test_policy(fn_random_policy, game)
        self.services.fn_record(f'Total Time Taken = {time() - self.start_time} seconds')
        return self

    @encapsulate
    def fn_pit_against_human(self):
        game = Game(self.services.args.board_size)
        fn_human_policy = fn_get_human_policy(game).fn_play_it
        self.fn_test_policy(fn_human_policy, game, verbose= True, num_of_games= 2)
        self.services.fn_record(f'Total Time Taken = {time() - self.start_time} seconds')
        return self

    @encapsulate
    def fn_archive_it(self):
        current_time_id = dt.now().strftime("%Y_%m_%d_%H_%M_%S")

        archive_path = os.path.join(self.services.archive_path, current_time_id)
        shutil.copytree(self.services.persister.log_root, archive_path, symlinks=False, ignore=None)

        return self

    @encapsulate
    def fn_clean(self):
        self.services.persister.fn_cleanup()
        return self

    @encapsulate
    def fn_create_new_session(self):
        session_id = self.services.persister.fn_create_new_session()
        self.services.fn_record()
        self.services.fn_record(f'SESSION ID: {session_id}')
        return self

    @encapsulate
    def fn_change_configs(self, configs):
        if configs is not None:
            for k,v in configs.items():
                if k in self.config_params.nnet_params:
                    self.config_params.nnet_params[k] = v
                    self.services.fn_record(f'{k}={v}')
        return self


    @staticmethod
    def fn_get_func_name():
        import traceback
        return traceback.extract_stack(None, 2)[0][2]

    @classmethod
    def fn_init(cls, file_path, config_params):
        agent = Agent(config_params, file_path)
        return agent

