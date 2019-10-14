
from maml_regression import MAML_Regression
from maml_regression import sample_data
import torch
import matplotlib.pyplot as plt

maml = MAML_Regression()

plt.ion()
plt.figure(2)

for itr in range(300):
    train_task_x, train_task_y, train_amplitude, train_phases = sample_data(5, 100)
    test_task_x, test_task_y, _, __ = sample_data(5, 10, train_amplitude, train_phases)
    test_x, test_y, test_amp, test_pha = sample_data(1, 10)
    meta_test_x, meta_test_y, _, __ = sample_data(1, 100, test_amp, test_pha)
    train_x, train_y = torch.tensor(train_task_x, dtype=torch.float32), torch.tensor(train_task_y,
                                                                                     dtype=torch.float32)
    test_task_x, test_task_y = torch.tensor(test_task_x, dtype=torch.float32), torch.tensor(test_task_y,
                                                                                            dtype=torch.float32)
    test_x, test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)
    meta_test_x, meta_test_y = torch.tensor(meta_test_x, dtype=torch.float32), torch.tensor(meta_test_y,
                                                                                          dtype=torch.float32)
    maml.baseline(train_x, train_y, test_x, test_y, meta_test_x, meta_test_y)
