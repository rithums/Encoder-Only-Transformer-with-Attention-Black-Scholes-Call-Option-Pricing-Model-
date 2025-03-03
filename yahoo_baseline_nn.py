import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloaders import SyntheticDataset, YahooDataset

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 5
        self.hidden_1 = nn.Linear(self.input_size, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.hidden_3 = nn.Linear(256, 64)
        self.hidden_4 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.hidden_1(x)
        out = F.relu(out)
        out = self.hidden_2(out)
        out = F.relu(out)
        out = self.hidden_3(out)
        out = F.relu(out)
        out = self.hidden_4(out)
        return out

model = Baseline()

criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 100
batch_size = 128

seed = torch.Generator().manual_seed(20)

train_data, other = torch.utils.data.random_split(YahooDataset(), [0.7,0.3])#, generator=seed)
val_data, test_data = torch.utils.data.random_split(other, [2/3, 1/3])#, generator=seed)

train_loader = DataLoader(dataset=train_data,batch_size=batch_size)
val_loader = DataLoader(dataset=val_data)
test_loader = DataLoader(dataset=test_data)

train_losses = []
val_losses = []

for e in range(num_epochs):

    model.train()

    for batch, labels in train_loader:

        optimizer.zero_grad()

        pred = model(batch)
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()
 

    with torch.no_grad():

        model.eval()

        train_x, train_y = train_data[:]
        train_pred = model(train_x)
        train_loss = criterion(train_pred, train_y).item()
        train_losses.append(train_loss)

        val_x, val_y = val_data[:]
        val_pred = model(val_x)
        val_loss = criterion(val_pred, val_y).item()
        val_losses.append(val_loss)

        print("Epoch " + str(e) + ", Train Loss=" + str(train_loss) + ", Val Loss=" + str(val_loss))

    np.save("results/yahoo_baseline_train_loss.npy", train_losses)
    np.save("results/yahoo_baseline_val_loss.npy", val_losses)
    



