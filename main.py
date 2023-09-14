import random
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.autograd.set_detect_anomaly(True)


class RandomTimeSeriesWindows(Dataset):
    def __init__(self, size, window_size):
        self.size = size
        self.window_size = window_size

    def __len__(self):
        return self.size
    
    def __getitem__(self, _):
        start = random.uniform(0, 2*torch.pi)
        return torch.linspace(start, random.randrange(3, 10)*torch.pi, self.window_size).sin()


class Model(nn.Module):
    def __init__(self, window_size):
        super().__init__()

        hidden_size = 8
        self.m = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, window_size)
        )
        self.a = nn.Sigmoid()

    def forward(self, x):
        x = self.m(x)
        x = self.a(x)
        x.data[:, 0] = 1
        x.data[:, -1] = 1
        return x
    

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0

    for x in dataloader:
        output = model(x)
        mask = output.round()
        output = x*mask

        # fig, axs = plt.subplots(4, figsize=(10, 6))
        # for i in range(4):
        #     axs[i].set_ylim(-1, 1)

        for i in range(output.shape[0]):
            # axs[0].plot(input[i].detach())
            # axs[0].set_title("input")
            # axs[1].plot(mask[i].detach())
            # axs[1].set_title("mask")
            # axs[2].plot(output[i].detach())
            # axs[2].set_title("output masked")

            for j in range(output.shape[1]):
                if mask[i, j] == 1:
                    left_index = j
                    left_value = output[i, j]
                else:
                    # we know that we have a almost right corner with 1
                    # so we dont need any checks here to avoid index overflow
                    for k in range(j+1, output.shape[1]):
                        if mask[i, k] == 1:
                            right_index = k
                            right_value = output[i, k]

                            # stop after finding right neighbor
                            break

                    b = left_value
                    m = (right_value-left_value) / (right_index-left_index)

                    current_delta = j-left_index

                    # new value
                    output[i, j] = m*current_delta + b

            # axs[3].plot(output[i].detach())
            # axs[3].set_title("output interpolated")
            # plt.show()

        alpha = 0.1
        loss = loss_fn(x, output)
        loss = loss + mask*alpha
        loss = loss.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
    return running_loss/len(dataloader)


WINDOW_SIZE = 8
BATCH_SIZE = 16
EPOCHS = 100

train_data = RandomTimeSeriesWindows(100, WINDOW_SIZE)
test_data = RandomTimeSeriesWindows(20, WINDOW_SIZE)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
model = Model(WINDOW_SIZE)
loss_fn = nn.MSELoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(EPOCHS):
    loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    print(f"{(epoch+1):2}: loss={loss:.4f}")








