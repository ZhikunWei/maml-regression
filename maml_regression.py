import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn.functional as F


def loss_mse(v1, v2):
    result = 0
    for a, b in zip(v1, v2):
        result += (a - b) ** 2
    return result / len(v1)


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


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, g):
        self.t += 1
        lr = self.lr * (1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        return lr * self.m / (self.v ** 0.5 + self.epsilon)


class MAML_Regression:
    def __init__(self, plot_fig=False):
        self.plot_figure = plot_fig
        self.weights = {'w1': torch.randn(1, 40, requires_grad=True), 'b1': torch.randn(1, 40, requires_grad=True),
                        'w2': torch.randn(40, 40, requires_grad=True), 'b2': torch.randn(1, 40, requires_grad=True),
                        'w3': torch.randn(40, 1, requires_grad=True), 'b3': torch.randn(1, 1, requires_grad=True)}
        self.num_update_alpha = 100
        self.num_update_beta = 10
        self.learning_rate_alpha = 0.0003
        self.learning_rate_beta = 0.0002
        self.meta_batch_size = 10
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = {'w1': torch.zeros(1, 40), 'b1': torch.zeros(1, 40), 'w2': torch.zeros(40, 40),
                  'b2': torch.zeros(1, 40), 'w3': torch.zeros(40, 1), 'b3': torch.zeros(1, 1)}
        self.v = {'w1': torch.zeros(1, 40), 'b1': torch.zeros(1, 40), 'w2': torch.zeros(40, 40),
                  'b2': torch.zeros(1, 40), 'w3': torch.zeros(40, 1), 'b3': torch.zeros(1, 1)}

        self.baseline_weights = {'w1': torch.randn(1, 40, requires_grad=True),
                                 'b1': torch.randn(1, 40, requires_grad=True),
                                 'w2': torch.randn(40, 40, requires_grad=True),
                                 'b2': torch.randn(1, 40, requires_grad=True),
                                 'w3': torch.randn(40, 1, requires_grad=True),
                                 'b3': torch.randn(1, 1, requires_grad=True)}

    def forward(self, weights, input_datas):
        outputs = []
        for input_data in input_datas:
            hidden1 = F.relu(torch.mm(input_data, weights['w1']) + weights['b1'])
            hidden2 = F.relu(torch.mm(hidden1, weights['w2']) + weights['b2'])
            output = torch.mm(hidden2, weights['w3']) + weights['b3']
            outputs.append(output)
        return outputs

    def meta_learning(self, input_datas, targets):
        fast_weights = {key: self.weights[key].clone().detach() for key in self.weights}

        for i in range(self.num_update_alpha):
            loss_all = 0
            for batch_index in range(int(len(input_datas) / self.meta_batch_size)):
                batch_input = input_datas[batch_index * self.meta_batch_size:(batch_index + 1) * self.meta_batch_size]
                batch_target = targets[batch_index * self.meta_batch_size:(batch_index + 1) * self.meta_batch_size]
                fast_weights = {key: fast_weights[key].requires_grad_(True) for key in fast_weights}
                for key in fast_weights:
                    try:
                        fast_weights[key].grad.data.zero_()
                    except:
                        pass
                predicts = self.forward(fast_weights, batch_input)
                loss2 = loss_mse(predicts, batch_target)
                loss2.backward()
                loss_all += loss2
                gradients = {key: fast_weights[key].grad for key in fast_weights}
                with torch.no_grad():
                    fast_weights = {key: fast_weights[key] - self.learning_rate_alpha * gradients[key] for key in
                                    fast_weights}

            with torch.no_grad():
                if self.plot_figure and i == self.num_update_alpha-1:
                    x = input_datas.data.numpy()
                    y_true = targets.data.numpy()
                    y_pred = [x.data.numpy() for x in self.forward(fast_weights, input_datas)]
                    ax1 = plt.subplot(4, 1, 1)
                    plt.cla()
                    ax1.set_title('meta training alpha %d epoch' % i)
                    l1 = plt.scatter(x, y_true, marker='.', c='b')
                    l2 = plt.scatter(x, y_pred, marker='.', c='r')
                    plt.legend((l1, l2), ("true", "predict"))
                    plt.pause(0.01)
        return fast_weights

    def meta_training(self, tasks_input, tasks_target, test_task_x, test_task_y):
        total_gradients = {'w1': torch.zeros(1, 40), 'b1': torch.zeros(1, 40),
                           'w2': torch.zeros(40, 40), 'b2': torch.zeros(1, 40),
                           'w3': torch.zeros(40, 1), 'b3': torch.zeros(1, 1)}
        for task_input, task_target, test_input, test_target in zip(tasks_input, tasks_target, test_task_x,
                                                                    test_task_y):
            task_weights = self.meta_learning(task_input, task_target)  # theta'
            task_weights = {key: task_weights[key].requires_grad_(True) for key in task_weights}
            try:
                task_weights = {key: task_weights[key].grad.data.zero_() for key in task_weights}
            except:
                pass
            task_predict = self.forward(task_weights, test_input)
            task_loss = loss_mse(task_predict, test_target)
            task_loss.backward()
            task_gradients = {key: task_weights[key].grad for key in task_weights}
            for key in total_gradients:
                total_gradients[key] = total_gradients[key] + task_gradients[key]
            with torch.no_grad():
                if self.plot_figure:
                    x = test_input.data.numpy()
                    y_true = test_target.data.numpy()
                    y_pred = [x.data.numpy() for x in task_predict]
                    ax1 = plt.subplot(4, 1, 1)
                    plt.cla()
                    ax1.set_title('meta training alpha')
                    l1 = plt.scatter(x, y_true, marker='.', c='b')
                    l2 = plt.scatter(x, y_pred, marker='.', c='r')
                    plt.legend((l1, l2), ("true", "predict"))
                # plt.pause(1)

        with torch.no_grad():
            self.t += 1
            for key in self.weights:
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * total_gradients[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * total_gradients[key] * total_gradients[key]
                m = self.m[key] / (1 - self.beta1 ** self.t)
                v = self.v[key] / (1 - self.beta2 ** self.t)
                self.weights[key] = self.weights[key] - self.learning_rate_beta * m / (v**0.5 + self.epsilon)
            if self.plot_figure:
                pred_after = self.forward(self.weights, tasks_input[0])
                x = tasks_input[0].data.numpy()
                y_true = tasks_target[0].data.numpy()
                y_pred_after = [x.data.numpy() for x in pred_after]
                ax1 = plt.subplot(4, 1, 2)
                plt.cla()
                ax1.set_title('meta training beta')
                l1 = plt.scatter(x, y_true, marker='.', c='b')
                l3 = plt.scatter(x, y_pred_after, marker='.', c='r')
                plt.legend((l1, l3), ("true", "after beta update"))
                plt.pause(1)

    def meta_testing(self, new_task_inputs, new_task_targets, new_task_test_inputs, new_task_test_targets):
        test_weights = {key: self.weights[key].clone().detach() for key in self.weights}
        for meta_test_input, meta_test_target in zip(new_task_test_inputs, new_task_test_targets):
            with torch.no_grad():
                final_pred = self.forward(test_weights, meta_test_input)
                final_loss = loss_mse(final_pred, meta_test_target)
                print("new task test loss", final_loss)
                if self.plot_figure:
                    x = meta_test_input.data.numpy()
                    y_true = meta_test_target.data.numpy()
                    y_pred = [x.data.numpy() for x in final_pred]
                    ax1 = plt.subplot(4, 1, 2)
                    plt.cla()
                    ax1.set_title('new task test')
                    l1 = plt.scatter(x, y_true, marker='.', c='b')
                    l2 = plt.scatter(x, y_pred, marker='.', c='r')
                    plt.legend((l1, l2), ("true", "predict"))
                    plt.pause(1)

        for new_input, new_target in zip(new_task_inputs, new_task_targets):
            for i in range(self.num_update_beta):
                test_weights = {key: test_weights[key].requires_grad_(True) for key in test_weights}
                for key in test_weights:
                    try:
                        test_weights[key].grad.data.zero_()
                    except:
                        pass
                new_task_pred = self.forward(test_weights, new_input)
                new_task_loss = loss_mse(new_task_pred, new_target)
                new_task_loss.backward()
                print("new task training loss", i, new_task_loss)
                # print('weights and gradient after backward', self.weights['b1'], self.weights['b1'].grad)
                new_task_gradients = {key: test_weights[key].grad for key in test_weights}
                with torch.no_grad():
                    for key in test_weights:
                        test_weights[key] = test_weights[key] - self.learning_rate_beta * new_task_gradients[key]
                    if self.plot_figure and i == self.num_update_beta-1:
                        new_task_predict = self.forward(test_weights, new_input)
                        x = new_input.data.numpy()
                        y_true = new_target.data.numpy()
                        y_pred = [x.data.numpy() for x in new_task_predict]
                        ax1 = plt.subplot(4, 1, 3)
                        plt.cla()
                        ax1.set_title('new task training')
                        l1 = plt.scatter(x, y_true, marker='.', c='b')
                        l2 = plt.scatter(x, y_pred, marker='.', c='r')
                        plt.legend((l1, l2), ("true", "predict"))
                        plt.pause(1)

        for meta_test_input, meta_test_target in zip(new_task_test_inputs, new_task_test_targets):
            with torch.no_grad():
                final_pred = self.forward(test_weights, meta_test_input)
                final_loss = loss_mse(final_pred, meta_test_target)
                print("new task test loss", final_loss)
                if self.plot_figure:
                    x = meta_test_input.data.numpy()
                    y_true = meta_test_target.data.numpy()
                    y_pred = [x.data.numpy() for x in final_pred]
                    ax1 = plt.subplot(4, 1, 4)
                    plt.cla()
                    ax1.set_title('new task test')
                    l1 = plt.scatter(x, y_true, marker='.', c='b')
                    l2 = plt.scatter(x, y_pred, marker='.', c='r')
                    plt.legend((l1, l2), ("true", "predict"))
                    plt.pause(1)

    def baseline(self, train_inputs, train_targets, new_task_inputs, new_task_targets, test_inputs, test_targets):
        for train_input, train_target in zip(train_inputs, train_targets):
            for i in range(self.num_update_alpha):
                self.baseline_weights = {key: self.baseline_weights[key].requires_grad_(True) for key in
                                         self.baseline_weights}
                try:
                    self.baseline_weights = {key: self.baseline_weights[key].grad.data.zero_() for key in
                                             self.baseline_weights}
                except:
                    pass
                baseline_train_pred = self.forward(self.baseline_weights, train_input)
                baseline_train_loss = loss_mse(train_target, baseline_train_pred)
                baseline_train_loss.backward()
                with torch.no_grad():
                    self.baseline_weights = {key: self.baseline_weights[key] - self.learning_rate_alpha *
                                                  self.baseline_weights[key].grad for key in self.baseline_weights}
                    print(i, 'baseline train loss', baseline_train_loss)
                    x = train_input.data.numpy()
                    y_true = train_target.data.numpy()
                    y_pred = [x.data.numpy() for x in baseline_train_pred]
                    plt.subplot(3, 1, 1)
                    plt.cla()
                    l1 = plt.scatter(x, y_true, marker='.', c='b')
                    l2 = plt.scatter(x, y_pred, marker='.', c='r')
                    plt.legend((l1, l2), ("true", "predict"))
                    plt.pause(0.1)

        for new_task_input, new_task_target in zip(new_task_inputs, new_task_targets):
            for i in range(self.num_update_beta):
                self.baseline_weights = {key: self.baseline_weights[key].requires_grad_(True) for key in
                                         self.baseline_weights}
                try:
                    self.baseline_weights = {key: self.baseline_weights[key].grad.data.zero_() for key in
                                             self.baseline_weights}
                except:
                    pass
                baseline_train_pred = self.forward(self.baseline_weights, new_task_input)
                baseline_train_loss = loss_mse(new_task_target, baseline_train_pred)
                baseline_train_loss.backward()
                with torch.no_grad():
                    self.baseline_weights = {key: self.baseline_weights[key] - self.learning_rate_beta *
                                                  self.baseline_weights[key].grad for key in self.baseline_weights}
                    print('baseline new task train loss', baseline_train_loss)
                    x = new_task_input.data.numpy()
                    y_true = new_task_target.data.numpy()
                    y_pred = [x.data.numpy() for x in baseline_train_pred]
                    plt.subplot(3, 1, 2)
                    plt.cla()
                    l1 = plt.scatter(x, y_true, marker='.', c='b')
                    l2 = plt.scatter(x, y_pred, marker='.', c='r')
                    plt.legend((l1, l2), ("true", "predict"))
                    plt.pause(1)

        for test_input, test_target in zip(test_inputs, test_targets):
            baseline_test_pred = self.forward(self.baseline_weights, test_input)
            baseline_test_loss = loss_mse(test_target, baseline_test_pred)
            print('baseline test loss', baseline_test_loss)
            with torch.no_grad():
                x = test_input.data.numpy()
                y_true = test_target.data.numpy()
                y_pred = [x.data.numpy() for x in baseline_test_pred]
                plt.subplot(3, 1, 3)
                plt.cla()
                l1 = plt.scatter(x, y_true, marker='.', c='b')
                l2 = plt.scatter(x, y_pred, marker='.', c='r')
                plt.legend((l1, l2), ("true", "predict"))
                plt.pause(1)


if __name__ == '__main__':
    plot_figure = False
    maml = MAML_Regression(plot_figure)

    if plot_figure:
        plt.ion()
        plt.figure(1)
    for itr in range(3000):
        maml.plot_figure = True
        train_task_x, train_task_y, train_amplitude, train_phases = sample_data(5, 100)
        test_task_x, test_task_y, _, __ = sample_data(5, 10, train_amplitude, train_phases)
        train_task_x, train_task_y = torch.tensor(train_task_x, dtype=torch.float32), torch.tensor(train_task_y,
                                                                                                   dtype=torch.float32)
        test_task_x, test_task_y = torch.tensor(test_task_x, dtype=torch.float32), torch.tensor(test_task_y,
                                                                                                dtype=torch.float32)
        maml.meta_training(train_task_x, train_task_y, test_task_x, test_task_y)

        new_task_x, new_task_y, test_amp, test_pha = sample_data(1, 10)
        new_task_test_x, new_task_test_y, _, __ = sample_data(1, 100, test_amp, test_pha)
        new_task_x, new_task_y = torch.tensor(new_task_x, dtype=torch.float32), torch.tensor(new_task_y,
                                                                                             dtype=torch.float32)
        new_task_test_x, new_task_test_y = torch.tensor(new_task_test_x, dtype=torch.float32), torch.tensor(
            new_task_test_y,
            dtype=torch.float32)
        maml.meta_testing(new_task_x, new_task_y, new_task_test_x, new_task_test_y)

        if itr % 500 == 498:
            maml.plot_figure = True
            with open('log/itr%d.pkl' % itr, 'wb') as f:
                pickle.dump(maml, f)
                print("save model of %d iteration" % itr)
