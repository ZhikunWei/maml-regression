import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn.functional as F


def sample_data(task_num, sample_per_task, batch_size, amplitude=None, phases=None):
    dataset = []
    X = []
    targets = []
    sample_x = np.random.uniform(-5, 5, [task_num, sample_per_task])
    sample_y = np.zeros([task_num, sample_per_task])
    if amplitude is None and phases is None:
        amplitude = np.random.uniform(0.5, 10, task_num)
        phases = np.random.uniform(0, np.pi, task_num)
    for i in range(len(sample_x)):
        for j in range(len(sample_x[i])):
            sample_y[i][j] = y = amplitude[i] * np.sin(sample_x[i][j] - phases[i])
            X.append(sample_x[i][j])
            targets.append(y)
            dataset.append((sample_x[i][j], y))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, sample_x, sample_y, amplitude, phases


train_loader, train_x, train_y, train_amplitude, train_phases = sample_data(
    task_num=1, sample_per_task=200, batch_size=1)
test_loader, test_x, test_y, _, __ = sample_data(
    task_num=1, sample_per_task=20, batch_size=1, amplitude=train_amplitude, phases=train_phases)


# for i in range(train_task_number):
#     plt.scatter(train_x[i], train_y[i])
# plt.show()


class NetRegreesion(torch.nn.Module):
    def __init__(self):
        super(NetRegreesion, self).__init__()
        self.linear1 = torch.nn.Linear(1, 40)
        self.linear2 = torch.nn.Linear(40, 40)
        self.linear3 = torch.nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = self.linear3(x)
        return y_pred


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
maml_net = NetRegreesion()
maml_net.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(maml_net.parameters(), lr=0.01)
plt.ion()
plt.figure()
for epoch in range(300):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)
        optimizer.zero_grad()
        y_pred = maml_net(inputs)
        loss = loss_fn(y_pred, labels)
        print(y_pred)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    with torch.no_grad():
        test_x = []
        y_true = []
        y_pred = []
        for data in train_loader:
            x, y = data[0].to(device), data[1].to(device)
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            output = maml_net(x)
            # print(x, x.item())
            test_x.append(x.item())
            y_true.append(y.item())
            y_pred.append(output.item())
        ax1 = plt.subplot(2, 1, 1)
        plt.cla()
        plt.scatter(test_x, y_true)
        plt.scatter(test_x, y_pred)
        # plt.pause(0.1)

        test_x = []
        y_true = []
        y_pred = []
        for data in test_loader:
            x, y = data[0].to(device), data[1].to(device)
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            output = maml_net(x)
            # print(x, x.item())
            test_x.append(x.item())
            y_true.append(y.item())
            y_pred.append(output.item())
        ax2 = plt.subplot(2, 1, 2)
        plt.cla()
        plt.scatter(test_x, y_true)
        plt.scatter(test_x, y_pred)

        plt.pause(0.1)

        print('[%d] loss: %.3f' %
              (epoch + 1, loss.item()))  # running_loss / len(train_loader)))
