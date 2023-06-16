from itertools import product
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import  SubsetRandomSampler
# Fix random
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
parameters = {
    'lr': [0.01, 0.001],
    'batch_size': [32, 64, 128],
    'shuffle': [True, False]
}
# param_values = [v for v in parameters.values()]
# print(param_values)
print(f"Using {device} device")

F = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,),)])
trainData = torchvision.datasets.FashionMNIST(
    root='./', train=True, download=False, transform=F)
testData = torchvision.datasets.FashionMNIST(
    root='./', train=False, download=False, transform=F)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28 ** 2, 10)
        self.fc1.weight.detach().zero_()
        self.fc1.bias.detach().zero_()
    def forward(self, x: torch.tensor):
        x = x.reshape(x.shape[0], 28 ** 2)
        x = self.fc1(x)
        yhat = nn.functional.softmax(x, dim = 1)
        return yhat


epochs = 100
# for run_id, (lr, batch_size, shuffle) in enumerate(product(*param_values)):
#     print("run id:", run_id + 1)
# print(f'batch_size = {batch_size} lr = {lr} shuffle = {shuffle}')
# idxs = np.random.permutation(len(trainData)) - 1
# train_sample = SubsetRandomSampler(indices=idxs[:1000])
model = Model().to(device)


optimizer = opt.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(trainData, batch_size=64, shuffle=True)
lossHist = []
for i in range(epochs):
    curLoss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        # print("s =", yhat.shape)
        loss = criterion(yhat, y)
        # print(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        curLoss += loss.item()*x.size(0)
    lossHist.append(curLoss)
    if i % 100 == 0:
        print(f"in epoch {i + 1}, loss = {curLoss}")
plt.plot(np.arange(1, epochs + 1), lossHist)
plt.show()
torch.save(model.state_dict(), 'model.pt')
