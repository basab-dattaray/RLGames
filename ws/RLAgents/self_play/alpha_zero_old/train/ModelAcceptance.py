from time import sleep

class ModelAcceptance():
    def __init__(self, services):

        self.services = services

        self.win_ratio_reduction_rate = .1
        if 'win_ratio_reduction_rate' in self.services.nnet_params:
            self.win_ratio_reduction_rate = services.nnet_params.win_ratio_reduction_rate

        msg, params = services.persister.fn_load_params()
        if msg is None and 'model_acceptance_win_ratio' in params.keys():
            self.model_acceptance_win_ratio = params['model_acceptance_win_ratio']
        else:
            self.model_acceptance_win_ratio = 0.6
            if 'model_acceptance_win_ratio' in self.services.nnet_params:
                self.model_acceptance_win_ratio = services.nnet_params.model_acceptance_win_ratio

        self.model_acceptance_ending_win_ratio = 0.5

        self.comparison_acceptance_ratio_iter = self.__fn_get_comparison_acceptance_ratio()

        self.comparison_acceptance_ratio = None

    def __fn_get_comparison_acceptance_ratio(self):
        iter_count = 0
        while True:
            numerator =  self.model_acceptance_ending_win_ratio * iter_count + self.model_acceptance_win_ratio / self.win_ratio_reduction_rate
            denominator = iter_count + 1/ self.win_ratio_reduction_rate
            comparison_acceptance_ratio  = numerator/denominator
            yield comparison_acceptance_ratio
            iter_count += 1

    def fn_do_accept_model(self, prev_policy_wins, new_policy_wins, win_ratio):
        self.comparison_acceptance_ratio = next(self.comparison_acceptance_ratio_iter)
        if prev_policy_wins + new_policy_wins == 0 or win_ratio < self.comparison_acceptance_ratio:
            return False
        else:
            return True

    def fn_get_comparison_acceptance_ratio(self):
        return self.comparison_acceptance_ratio

