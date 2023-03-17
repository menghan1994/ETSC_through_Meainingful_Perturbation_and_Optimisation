from mlxtend.data import loadlocal_mnist
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("..") 
from datasets.mask import MaskGenerator

class MNIST(Dataset):
    def __init__(self):
        self.train_X, self.train_y = loadlocal_mnist(
        images_path='./datasets/Data/MNIST/raw/train-images-idx3-ubyte', 
        labels_path='./datasets/Data/MNIST/raw/train-labels-idx1-ubyte'
        )

        self.test_X, self.test_y = loadlocal_mnist(
        images_path='./datasets/Data/MNIST/raw/t10k-images-idx3-ubyte', 
        labels_path='./datasets/Data/MNIST/raw/t10k-labels-idx1-ubyte'
        )

        self.train_X = (self.train_X/255.0 - 0.5)/0.5
        self.test_X = (self.test_X/255.0- 0.5)/0.5


        self.mask_generator = MaskGenerator((28, 28))

class MNIST_train(MNIST):
    def __init__(self):
        super().__init__()

    def __len__(self):

        return self.train_y.shape[0]

    def __getitem__(self, idx):


        x = self.train_X[idx].reshape(28, 28)
        mask, delta = self.mask_generator.mask(return_delta=True)
        # mask, delta = self.mask_generator.random_mask(return_delta=True)

        return {'values': x, 'masks': mask, 'deltas': delta, 'evals': x * (1-mask), 'eval_masks': (1-mask)}, self.train_y[idx]



class MNIST_test(MNIST):
    def __init__(self):
        super().__init__()


    def __len__(self):
        return self.test_y.shape[0]

    def __getitem__(self, idx):

        x = self.test_X[idx].reshape(28, 28)
        mask, delta = self.mask_generator.mask(return_delta=True)
        # mask, delta = self.mask_generator.random_mask(return_delta=True)

        return {'values': x, 'masks': mask, 'deltas': delta, 'evals': x * (1-mask), 'eval_masks': (1-mask)}, self.test_y[idx]


