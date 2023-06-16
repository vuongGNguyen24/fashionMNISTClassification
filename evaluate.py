from itertools import product
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28 ** 2, 10)
        self.fc1.weight.detach().zero_()
        self.fc1.bias.detach().zero_()

    def forward(self, x: torch.tensor):
        x = x.reshape(x.shape[0], 28 ** 2)
        x = self.fc1(x)
        yhat = nn.functional.softmax(x, dim=1)
        return yhat


print("in line 12")
F = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,),)])
testData = torchvision.datasets.FashionMNIST(
    root='./', train=False, download=False, transform=F)
model = Model()
model.load_state_dict(torch.load(
    r"D:\ml_exam\Fashion MNIST Classification\model.pt"))
class_correct = np.zeros(10)
class_total = np.zeros(10)
testLoader = DataLoader(testData, batch_size=1)

model.eval()
for x, y in testLoader:
    yHat = model(x)
    predict = torch.argmax(yHat)
    if predict == y:
        class_correct[y] += 1
    class_total[y] += 1
       
 


for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' %
              (str(i), 100 * class_correct[i]/class_total[i],
               np.sum(class_correct[i]), np.sum(class_total[i])))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
