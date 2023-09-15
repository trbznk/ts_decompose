import random
import torch
import argparse
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
    def __init__(self, window_size, max_marker):
        super().__init__()

        self.max_marker = max_marker

        hidden_size = 8
        self.m = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, window_size)
        )
        self.a = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.m(x)
        x = self.a(x)
        x.data[:, 0] = 1
        x.data[:, -1] = 1

        # NOTE: this is probably very slow @Speedup
        top_values, top_indices = torch.topk(x, self.max_marker)
        top_values = top_values.min(axis=1).values
        top_values = top_values.view(-1, 1)

        # TODO: maybe we dont need this here but its better to test it
        #       in a much simpler example
        # top_values = top_values.repeat(1, 100)

        x.data[x >= top_values] = 1
        x.data[x < top_values] = 0

        return x
    

def generate_decomp(x, mask):
    output = x*mask
    for i in range(output.shape[0]):
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
    return output


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    running_loss = 0.0
    for x in dataloader:
        mask = model(x)
        output = generate_decomp(x, mask)

        loss = loss_fn(x, output)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
    return running_loss/len(dataloader)


def test_loop(dataloader, model, loss_fn):
    model.eval()

    with torch.no_grad():
        for x in dataloader:
            mask = model(x)
            output = generate_decomp(x, mask)

            loss = loss_fn(x, output)
            
            fig, ax = plt.subplots(1, figsize=(10, 6))
            ax.set_ylim(-1.5, 1.5)

            window = x.view(-1)
            window_decomposed = output.view(-1)
            amount_markers = int(mask.sum().item())
            markers = (x*mask).view(-1)
            markers[markers == 0] = float("nan")

            ax.set_title(f"markers={amount_markers}")
            ax.plot(window)
            ax.plot(window_decomposed)
            ax.plot(markers, "k.")
            plt.show()


if __name__ == "__main__":
    WINDOW_SIZE = 100
    BATCH_SIZE = 16
    EPOCHS = 5
    MAX_MARKER = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-p", "--predict", action="store_true")
    args = parser.parse_args()

    train_data = RandomTimeSeriesWindows(100, WINDOW_SIZE)
    test_data = RandomTimeSeriesWindows(5, WINDOW_SIZE)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=1)
    model = Model(WINDOW_SIZE, MAX_MARKER)
    loss_fn = nn.MSELoss()

    if args.train:    
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(EPOCHS):
            loss = train_loop(train_dataloader, model, loss_fn, optimizer)
            print(f"{(epoch+1):2}: loss={loss:.4f}")

        torch.save(model, "model.pth")

    if args.predict:
        model = torch.load("model.pth")
        loss = test_loop(test_dataloader, model, loss_fn)

    if not args.predict and not args.train:
        parser.print_help()
