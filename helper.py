import numpy as np
class batch_generator():
    def __init__(self, idata, batch_size):
        """
            idata = list of pairs of indexed sentence data and sentiment value
            batch_size = the batch size to be used by the genrator
        """
        self.idata = idata
        self.batch_size = batch_size
        self.current_idx = 0
        
    def batch(self):
        """
            Creates a batch in time major format i.e [max_time, batch_size] along with array containing sentiment value.
            The batch creation is dynamic as in the given batch size the sentences are padded only to length of max
            length sentencs in batch
        """
        while True:
            if self.current_idx + self.batch_size > len(self.idata):
                self.current_idx = 0
            work_batch = self.idata[self.current_idx : self.current_idx + self.batch_size]
            self.current_idx += self.batch_size
            seq_lens = [len(seq[0]) for seq in work_batch]
            max_seq_len = max(seq_lens)
            inputs_batch = np.zeros([self.batch_size, max_seq_len])
            inputs_label = np.zeros([self.batch_size])
            for i, pair in enumerate(work_batch):
                for j, sent in enumerate(pair[0]):
                    inputs_batch[i, j] = sent
                inputs_label[i] = pair[1]
            #inputs_time_major = inputs_batch.swapaxes(0, 1)
            yield inputs_batch, inputs_label
            