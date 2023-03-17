import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append("..") 
from datasets.mask import MaskGenerator

class MultivariateUEA(Dataset):
    def __init__(self, problem_name, MTS_size):
        self.problem_name = problem_name
        self.train_x = np.load(f'./datasets/Data/MultivariateUEA/{problem_name}/train_ts.npy')
        self.train_labels = np.load(f'./datasets/Data/MultivariateUEA/{problem_name}/train_labels.npy')

        self.test_x = np.load(f'./datasets/Data/MultivariateUEA/{problem_name}/test_ts.npy')
        self.test_labels = np.load(f'./datasets/Data/MultivariateUEA/{problem_name}/test_labels.npy')
        self.mts_sizt = MTS_size
        self.mask_generator = MaskGenerator((self.test_x.shape[1], self.test_x.shape[2]))


class MultivariateUEA_train(MultivariateUEA):
    def __init__(self, problem_name, MTS_size = None):
        super().__init__(problem_name, MTS_size = MTS_size)

    def __len__(self):

        return self.train_labels.shape[0]

    def __getitem__(self, idx):
            
        x = self.train_x[idx]
        mask, delta = self.mask_generator.mask(return_delta=True)
        # mask, delta = self.mask_generator.random_mask(return_delta=True)

        return {'values': x, 'masks': mask, 'deltas': delta, 'evals': x * (1-mask), 'eval_masks': (1-mask)}, self.train_labels[idx]

class MultivariateUEA_test(MultivariateUEA):
    def __init__(self, problem_name, MTS_size = None):
        super().__init__(problem_name, MTS_size = MTS_size)

    def __len__(self):
        return self.test_labels.shape[0]

    def __getitem__(self, idx):            
        x = self.test_x[idx]
        mask, delta = self.mask_generator.mask(return_delta=True)
        # mask, delta = self.mask_generator.random_mask(return_delta=True)

        return {'values': x, 'masks': mask, 'deltas': delta, 'evals': x * (1-mask), 'eval_masks': (1-mask)}, self.test_labels[idx]




