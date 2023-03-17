import numpy as np
import matplotlib.pyplot as plt

class MaskGenerator():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.feat_num = self.input_shape[1]
        self.seq_len = self.input_shape[0]
    
    def mask(self, return_delta = False):
        #select features to mask
        feat_ind = np.random.randint(0, self.feat_num, (self.feat_num, ))
        time_point1 = np.random.randint(0, self.seq_len, (self.feat_num,))
        time_point2 = np.random.randint(0, self.seq_len, (self.feat_num ,))

        mask = np.ones(self.input_shape)
        for mask_ind in range(len(feat_ind)):
            start_time_point, end_time_point = min(time_point1[mask_ind], time_point2[mask_ind]), max(time_point1[mask_ind], time_point2[mask_ind])
            mask[start_time_point:end_time_point, feat_ind[mask_ind]] = 0

        if return_delta:
            return mask, self.get_delta(mask)
        else:
            return mask

    def get_delta(self, mask):
        delta = np.ones_like(mask)
        delta[0, :] = 0
        feat = delta.shape[-1]
        seq = delta.shape[-2]
        for i in range(feat):
            for j in range(1, seq):
                if mask[j-1, i] == 0:
                    delta[j, i] = 1 + delta[j-1, i]
                else:
                    delta[j, i] = 1
        return delta


    def random_mask(self, return_delta = False):
        #select features to mask
        
        mask = np.ones(self.input_shape)
        num_zeros = int(mask.size * 0.2)
        zero_indices = np.random.choice(mask.size, num_zeros, replace=False)
        mask.flat[zero_indices] = 0

        if return_delta:
            return mask, self.get_delta(mask)
        else:
            return mask