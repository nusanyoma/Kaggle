from sklearn.model_selection import StratifiedKFold
import numpy as np


class StackedGeneralization:
    def __init__(self, n_splits, train_data, train_target, test_data):
        self.n_splits = n_splits
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.n_classes = len(np.unique(train_target))

        self.skf = StratifiedKFold(n_splits=n_splits)
