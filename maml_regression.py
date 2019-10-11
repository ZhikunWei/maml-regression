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


class MAML_Regression:
    def __init__(self):
        self.weights = {'w1': torch.randn(1, 40, requires_grad=True), 'b1': torch.randn(1, 40, requires_grad=True),
                        'w2': torch.randn(40, 40, requires_grad=True), 'b2': torch.randn(1, 40, requires_grad=True),
                        'w3': torch.randn(40, 1, requires_grad=True), 'b3': torch.randn(1, 1, requires_grad=True)}
        self.num_update = 5
        self.learning_rate_alpha = 0.0003
        self.learning_rate_beta = 0.0001

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
        predicts = self.forward(self.weights, input_datas)
        loss_a = loss_mse(predicts, targets)
        for key in self.weights:
            try:
                self.weights[key].grad.data.zero_()
            except:
                pass
        loss_a.backward()
        # print('loss init', loss_a.data)
        gradients = {key: self.weights[key].grad for key in self.weights}
        fast_weights = {key: self.weights[key].clone().detach() - self.learning_rate_alpha * gradients[key]
                        for key in self.weights}
        for i in range(self.num_update):
            fast_weights = {key: fast_weights[key].requires_grad_(True) for key in fast_weights}
            for key in fast_weights:
                try:
                    fast_weights[key].grad.data.zero_()
                except:
                    pass
            predicts = self.forward(fast_weights, input_datas)
            loss2 = loss_mse(predicts, targets)
            loss2.backward()
            gradients = {key: fast_weights[key].grad for key in fast_weights}
            with torch.no_grad():
                fast_weights = {key: fast_weights[key] - self.learning_rate_alpha * gradients[key] for key in
                                fast_weights}
            print('training loss', i, loss2)
        return fast_weights

    def meta_training_alpha(self, tasks_input, tasks_target):
        total_gradients = {'w1': torch.zeros(1, 40), 'b1': torch.zeros(1, 40),
                           'w2': torch.zeros(40, 40), 'b2': torch.zeros(1, 40),
                           'w3': torch.zeros(40, 1), 'b3': torch.zeros(1, 1)}
        for task_input, task_target in zip(tasks_input, tasks_target):
            task_weights = self.meta_learning(task_input, task_target)
            task_weights = {key: task_weights[key].requires_grad_(True) for key in task_weights}
            for key in task_weights:
                try:
                    task_weights[key].grad.data.zero_()
                except:
                    pass
            task_predict = self.forward(task_weights, task_input)
            task_loss = loss_mse(task_predict, task_target)
            task_loss.backward()
            task_gradients = {key: task_weights[key].grad for key in task_weights}
            for key in total_gradients:
                total_gradients[key] = total_gradients[key] + task_gradients[key]
            with torch.no_grad():
                x = task_input.data.numpy()
                y_true = task_target.data.numpy()
                y_pred = [x.data.numpy() for x in task_predict]
                plt.subplot(4, 1, 1)
                plt.cla()
                plt.scatter(x, y_true, marker='.', c='b')
                plt.legend('true value')
                plt.scatter(x, y_pred, marker='.', c='r')
                plt.legend('predict value')
                plt.pause(1)

        with torch.no_grad():
            pred_before = self.forward(self.weights, tasks_input[0])
            loss_before = loss_mse(pred_before, tasks_target[0])
            for key in self.weights:
                self.weights[key] = self.weights[key] - self.learning_rate_beta * total_gradients[key]
            pred_after = self.forward(self.weights, tasks_input[0])
            loss_after = loss_mse(pred_after, tasks_target[0])
            print('loss before', loss_before, 'loss after', loss_after)


    def meta_testing_beta(self, new_task_inputs, new_task_targets):
        for new_input, new_target in zip(new_task_inputs, new_task_targets):
            self.weights = {key: self.weights[key].requires_grad_(True) for key in self.weights}
            for key in self.weights:
                try:
                    self.weights[key].grad.data.zero_()
                except:
                    pass
            new_task_pred = self.forward(self.weights, new_input)
            new_task_loss = loss_mse(new_task_pred, new_target)
            new_task_loss.backward()
            # print('weights and gradient after backward', self.weights['b1'], self.weights['b1'].grad)
            new_task_gradients = {key: self.weights[key].grad for key in self.weights}
            with torch.no_grad():
                for key in self.weights:
                    self.weights[key] = self.weights[key] - self.learning_rate_beta * new_task_gradients[key]
                new_task_predict = self.forward(self.weights, new_input)
                loss_after_training = loss_mse(new_task_predict, new_target)
                print("loss before training =", new_task_loss, ", loss after training =", loss_after_training)

    def final_test(self, meta_test_inputs, meta_test_targets):
        for meta_test_input, meta_test_target in zip(meta_test_inputs, meta_test_targets):
            with torch.no_grad():
                final_pred = self.forward(self.weights, meta_test_input)
                final_loss = loss_mse(final_pred, meta_test_target)
                print('final loss', final_loss)

    def baseline(self, train_inputs, train_targets, new_task_inputs, new_task_targets, test_inputs, test_targets):
        for train_input, train_target in zip(train_inputs, train_targets):
            for i in range(self.num_update):
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
                    y_pred  = [x.data.numpy() for x in baseline_train_pred]
                    plt.subplot(3, 1, 1)
                    plt.cla()
                    plt.scatter(x, y_true, marker='.', c='b')
                    plt.legend('true value')
                    plt.scatter(x, y_pred, marker='.', c='r')
                    plt.legend('predict value')
                    plt.pause(1)

        for new_task_input, new_task_target in zip(new_task_inputs, new_task_targets):
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
                plt.scatter(x, y_true, marker='.', c='b')
                plt.legend('true value')
                plt.scatter(x, y_pred, marker='.', c='r')
                plt.legend('predict value')
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
                plt.scatter(x, y_true, marker='.', c='b')
                plt.legend('true value')
                plt.scatter(x, y_pred, marker='.', c='r')
                plt.legend('predict value')
                plt.pause(0)
        plt.show()

train_task_x, train_task_y, train_amplitude, train_phases = sample_data(5, 100)
test_x, test_y, test_amp, test_pha = sample_data(1, 10)
meta_test_x, meta_test_y, _, __ = sample_data(1, 100, test_amp, test_pha)
train_x, train_y = torch.tensor(train_task_x, dtype=torch.float32), torch.tensor(train_task_y, dtype=torch.float32)
test_x, test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)
meta_test_x, meta_test_y = torch.tensor(meta_test_x, dtype=torch.float32), torch.tensor(meta_test_y,
                                                                                        dtype=torch.float32)

plt.ion()

maml = MAML_Regression()
plt.figure()
maml.meta_training_alpha(train_x, train_y)
maml.meta_testing_beta(test_x, test_y)
maml.final_test(meta_test_x, meta_test_y)

plt.figure()
maml.baseline(train_x, train_y, test_x, test_y, meta_test_x, meta_test_y)
