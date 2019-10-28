import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy


def sample_data(task_num, sample_per_task, amplitude=None, phases=None):
    sample_x = np.random.uniform(-5, 5, [task_num, sample_per_task, 1, 1])
    sample_y = np.zeros([task_num, sample_per_task, 1, 1])
    if amplitude is None and phases is None:
        amplitude = np.random.uniform(0.1, 5, task_num)
        phases = np.random.uniform(0, np.pi, task_num)
    for i in range(len(sample_x)):
        for j in range(len(sample_x[i])):
            sample_y[i][j] = y = amplitude[i] * np.sin(sample_x[i][j] - phases[i])
    return sample_x, sample_y, amplitude, phases


class MyMAML(nn.Module):
    def __init__(self):
        super(MyMAML, self).__init__()
        self.linear1 = nn.Linear(1, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, 1)
        self.learning_rate = 0.01

        # Adam updater
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = [[torch.zeros(40, 1), torch.zeros(40, 1)], [torch.zeros(40, 40), torch.zeros(40, 40)],
                    [torch.zeros(1, 40), torch.zeros(1, 40)]]
        self.v = [[torch.zeros(40, 1), torch.zeros(40, 1)], [torch.zeros(40, 40), torch.zeros(40, 40)],
                    [torch.zeros(1, 40), torch.zeros(1, 40)]]

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def loss_mse(v1, v2):
    result = 0
    for a, b in zip(v1, v2):
        result += (a - b) ** 2
    return result / len(v1)


def meta_train(maml, inputs, targets):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(maml.parameters(), lr=0.01)

    for x, y in zip(inputs, targets):
        y = torch.tensor(y, dtype=torch.float32)
        y_pred = maml(x)
        loss = loss_fn(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def meta_test(maml, inputs, targets):
    loss = 0
    for x, y in zip(inputs, targets):
        y_pred = maml(x)
        loss += nn.MSELoss(y, y_pred)
    maml.zero_grad()
    loss.backward()


def meta_update(maml, maml_tmp):
    maml.t += 1
    with torch.no_grad():
        for layer, m, v, meta_layer in zip([maml.linear1, maml.linear2, maml.linear3], maml.m, maml.v,
                                                       [maml_tmp.linear1, maml_tmp.linear2, maml_tmp.linear3]):
            m[0] = maml.beta1 * m[0] + (1 - maml.beta1) * meta_layer.weight.grad
            v[0] = maml.beta2 * v[0] + (1 - maml.beta2) * meta_layer.weight.grad * meta_layer.weight.grad
            m[0] = m[0] / (1 - maml.beta1 ** maml.t)
            v[0] = v[0] / (1 - maml.beta2 ** maml.t)
            layer.weight.data = layer.weight.data - maml.learning_rate * m[0] / (v[0] ** 0.5 + maml.epsilon)

            m[1] = maml.beta1 * m[1] + (1 - maml.beta1) * meta_layer.bias.grad
            v[1] = maml.beta2 * v[1] + (1 - maml.beta2) * meta_layer.bias.grad * meta_layer.bias.grad
            m[1] = m[1] / (1 - maml.beta1 ** maml.t)
            v[1] = v[1] / (1 - maml.beta2 ** maml.t)
            layer.bias.data = layer.bias.data - maml.learning_rate * m[1] / (v[1] ** 0.5 + maml.epsilon)


def train():
    maml = MyMAML()
    for i in range(3000):
        tasks_inputs_train, tasks_targets_train, amp, pha = sample_data(5, 100)
        tasks_inputs_test, tasks_targets_test, a, aa = sample_data(5, 10, amp, pha)
        for j in range(len(tasks_inputs_train)):
            maml_tmp = copy.deepcopy(maml)
            meta_train(maml_tmp, tasks_inputs_train[j], tasks_targets_train[j])
            meta_test(maml_tmp, tasks_inputs_test[j], tasks_targets_test[j])
            meta_update(maml, maml_tmp)


if __name__ == '__main__':
    train()
