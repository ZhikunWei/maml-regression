import pickle

import torch
from maml_regression import MAML_Regression
from maml_regression import sample_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('log/itr2999.pkl', 'rb') as f:
        maml = pickle.load(f)
    maml.plot_figure = True
    new_task_x, new_task_y, test_amp, test_pha = sample_data(1, 10)
    new_task_test_x, new_task_test_y, _, __ = sample_data(1, 100, test_amp, test_pha)
    new_task_x, new_task_y = torch.tensor(new_task_x, dtype=torch.float32), torch.tensor(new_task_y,
                                                                                         dtype=torch.float32)
    new_task_test_x = torch.tensor(new_task_test_x, dtype=torch.float32)
    new_task_test_y = torch.tensor(new_task_test_y, dtype=torch.float32)

    maml.meta_testing(new_task_x, new_task_y, new_task_test_x, new_task_test_y)
    plt.show()
