from random import shuffle


class SampleRepository():
    def __init__(self, persister):
        self.samples_buffer = []
        self.persister = persister

    def fn_append_batch_of_samples(self, samples):
        samples_tuple = (samples, False)
        self.samples_buffer.append(samples_tuple)

    def fn_pop_a_batch_of_samples(self):

        return self.samples_buffer.pop(0)

    def fn_get_the_number_of_sample_batches(self):
        return len(self.samples_buffer)

    def fn_get_shuffled_samples(self):
        training_samples = []
        for e, x in self.samples_buffer:
            training_samples.extend(e)
        shuffle(training_samples)

        return training_samples

    # def fn_load(self):
    #     msg, self.samples_buffer = self.persister.fn_load_samples()
    #     if msg is not None:
    #         self.samples_buffer = []
    #     return msg

    def fn_get_num_of_samples(self):
        num_samples = 0
        for samples, qualifier in self.samples_buffer:
            num_samples += len(samples)
        return num_samples

    def fn_save_newer_samples(self):

        # find the latest entry - it should be as a result of ACCEPT
        latest_entry = None
        if len(self.samples_buffer) > 0:
            latest_entry =  (self.samples_buffer[-1][0], True)

        # Create new_samples_buffer by pruning all entries with REJECTS (False)
        new_samples_buffer = []
        for entry in self.samples_buffer:
            samples, qualifier = entry
            if qualifier is True: # False => REJECTS from previous iterations, include only ACCEPTS (True)
                new_samples_buffer.append(entry)

        #

        if latest_entry is not None:
            new_samples_buffer.append(latest_entry)
        else:
            exit() # cannot land here as fn_save is always called for ACCEPT

        self.samples_buffer = new_samples_buffer

        self.persister.fn_save_samples(self.samples_buffer)
        msg = f'@@@ Saved samples: {self.fn_get_the_number_of_sample_batches()} batches comprising {self.fn_get_num_of_samples()} samples'
        return msg