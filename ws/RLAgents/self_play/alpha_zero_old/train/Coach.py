
import os
import shutil


from ws.RLAgents.self_play.alpha_zero_old.play.Arena import Arena
from ws.RLAgents.self_play.alpha_zero_old.train.ModelAcceptance import ModelAcceptance
from ws.RLAgents.self_play.alpha_zero_old.train.SampleMgr import SampleMgr
from ws.RLAgents.self_play.alpha_zero_old.train.SampleRepository import SampleRepository
from ws.RLAgents.self_play.alpha_zero_old.train.SampleGenerator import SampleGenerator

from ws.RLInterfaces.PolicyTypes import PolicyTypes
from ws.RLUtils.monitoring.charting.Chart import Chart

class Coach():

    def __init__(self, services, nnet):
        self.services = services
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.services)

        self.skipping_first_pass_because_samples_are_loaded = False

        def fn_title_update_callback(progress_info):
            session_id = self.services.persister.fn_get_new_session_id()
            msg = '{}::  Iterations: ({}/{}), Model Uprades: ({}/{}), Epoch: ({}/{}), Batch: ({}/{})'.format(
                session_id,
                progress_info['iteration_index'], progress_info['max_iterations'],
                progress_info['new_model_count'], progress_info['max_model_upgrades'],
                progress_info['epoch_index'], progress_info['max_epochs'],
                progress_info['batch_index'], progress_info['max_batches']
            )
            return msg

        plot_file_path = os.path.join(self.services.persister.session_folder, 'loss_plot.pdf')
        self.services.chart = Chart(
            plot_file_path,
            title_prefix='Alpha Zero: Othello\n',
            fn_title_update_callback = fn_title_update_callback,
            x_config_item={'axis_label': 'event number'},
            y_config_list=[
                {'axis_label': 'total loss', 'color_black_background': 'green'},
                {'axis_label': 'value loss', 'color_black_background': 'orange'},
                {'axis_label': 'action probability loss', 'color_black_background': 'blue'}
            ],
            average_interval=self.services.args.mean_log_interval, skip_interval=self.services.args.skip_log_interval
        )
        self.sample_mgr = SampleMgr(services, nnet)
        self.model_acceptance = ModelAcceptance(services)

    def learn(self):
        self.fn_setup_for_training()

        upgraded_model_count = 0
        iter_index = 0
        max_iterations = self.services.args['num_iterations']
        while (upgraded_model_count < self.services.args['num_model_upgrades'] ) \
                and (iter_index < max_iterations):
            iter_index += 1

            self.services.fn_record()
            self.services.fn_record(f'SAMPLE GEN ITERATION {iter_index} of {max_iterations}')

            # save nnet then load pnet
            self.services.persister.fn_xfer_model_using_file(self.nnet, self.pnet)
            prev_system_policy = self.services.policy_broker.fn_get_best_action_policy(
                PolicyTypes.POLICY_TRAINING_EVAL,
                self.pnet
            )

            self.services.fn_record(f'@@@ Upgraded Model')

            iteration_info = {'max_iterations': self.services.args['num_iterations'], 'iteration_index': iter_index,
                              'max_model_upgrades': self.services.args['num_model_upgrades'], 'new_model_count': upgraded_model_count}

            training_samples = self.sample_mgr.fn_get_training_samples(iter_index, self.skipping_first_pass_because_samples_are_loaded)

            self.services.fn_record(f'@@@ Training the Model Nnet with {len(training_samples)} samples')
            self.nnet.train(training_samples, iteration_info, self.services.chart)

            next_system_policy = self.services.policy_broker.fn_get_best_action_policy(
                PolicyTypes.POLICY_TRAINING_EVAL,
                self.nnet
            )

            self.services.fn_record()
            self.services.fn_record('PITTING AGAINST PREVIOUS VERSION')

            arena = Arena(self.services,
                          self.services.game
                          )

            new_policy_wins, prev_policy_wins, draws = arena.playGames(
                next_system_policy,
                prev_system_policy,
                self.services.args.num_training_eval_games)

            self.services.fn_record('  NEW/PREV WINS : %d / %d ; DRAWS : %d' % (new_policy_wins, prev_policy_wins, draws))
            win_ratio = float(new_policy_wins) / (prev_policy_wins + new_policy_wins)

            if self.model_acceptance.fn_do_accept_model(prev_policy_wins, new_policy_wins, win_ratio):
                upgraded_model_count += 1
                self.services.fn_record(
                    f'  ACCEPTING NEW MODEL: win_ratio = {win_ratio} threshold: {self.model_acceptance.fn_get_comparison_acceptance_ratio()} Model Acceptance: {upgraded_model_count} of {iter_index}')
                self.fn_save_in_workspace()
            else:
                self.services.fn_record(f'  REJECTING NEW MODEL: win_ratio = {win_ratio} threshold: {self.model_acceptance.fn_get_comparison_acceptance_ratio()}, Model Acceptance: {upgraded_model_count} of {iter_index}')
                self.services.persister.fn_xfer_model_using_file(self.pnet, self.nnet)

        self.services.chart.fn_close()

        model_acceptance_win_ratio = self.model_acceptance.fn_get_comparison_acceptance_ratio()
        self.services.persister.fn_save_params({'model_acceptance_win_ratio': model_acceptance_win_ratio})

        if os.path.exists(self.services.persister.model_file_path):
            shutil.copy(self.services.persister.model_file_path, self.services.persister.session_folder)

        if os.path.exists(self.services.persister.samples_file_path):
            shutil.copy(self.services.persister.samples_file_path, self.services.persister.session_folder)

        if not self.services.persister.fn_model_exists():
            self.services.fn_record('*** WARNING: Model was not created')

    def fn_save_in_workspace(self):
        self.sample_mgr.fn_save()
        msg = self.services.persister.fn_save_model(self.nnet)
        if msg is not None:
            self.services.fn_record(msg)

    def fn_setup_for_training(self):
        # x = {'abc': 12.34}
        # self.services.persister.fn_save_params(x)
        # y = self.services.persister.fn_load_params()
        msg = self.services.persister.fn_load_model(self.nnet)
        if msg is not None:
            self.services.fn_record(msg)
        else:
            self.services.fn_record('@@@ Model loaded')

