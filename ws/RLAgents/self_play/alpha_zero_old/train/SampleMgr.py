from ws.RLAgents.self_play.alpha_zero_old.train.SampleGenerator import SampleGenerator
from ws.RLAgents.self_play.alpha_zero_old.train.SampleRepository import SampleRepository


class SampleMgr():
    def __init__(self, services, nnet):
        self.services = services
        self.sample_generator = SampleGenerator(services, nnet)
        self.sample_repo = SampleRepository(services.persister)
        self.nnet = nnet

    def fn_get_training_samples(self, iter_index, skipping_first_pass_because_samples_are_loaded):
        if not skipping_first_pass_because_samples_are_loaded or iter_index > 1:
            samples = self.sample_generator.fn_generate_samples()
            self.sample_repo.fn_append_batch_of_samples(samples)
        while self.sample_repo.fn_get_the_number_of_sample_batches() > self.services.args.num_sample_buffers:
            self.sample_repo.fn_pop_a_batch_of_samples()
        training_samples = self.sample_repo.fn_get_shuffled_samples()
        return training_samples

    def fn_save(self):
        msg = self.sample_repo.fn_save_newer_samples()
        self.services.fn_record(msg)

