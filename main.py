import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import Net

# check for gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# define a transformation pipeline for our raw data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download raw data and then apply transformation pipeline
print("downloading training data...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# then construct a generator from the transformed data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# instantiate a model according to that architecture
print("instantiating model...")
net = Net().to(device)  # send it to gpu
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


n_epochs = 100

print("training")
for epoch in range(n_epochs):

    for batch_idx, data in enumerate(trainloader):

        X, y = data[0].to(device), data[1].to(device)  # need to send data to gpu too, not just the model

        optimizer.zero_grad()  # clean up gradients
        outputs = net(X)  # emit class logits
        loss = criterion(outputs, y)  # compute loss
        loss.backward()  # compute gradients wrt loss
        optimizer.step()  # update weights according to computed gradients

        # print statistics
        if batch_idx % 500 == 0:
            print("epoch: %d loss: %.3f" % (epoch, loss.item()))
print('Finished Training')

print("downloading test data...")
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=True)

print("testing")
net.eval()
for data in testloader:
    
    X, y = data[0].to(device), data[1].to(device)
    
    output = net(X)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    correct = 0
    correct += pred.eq(y.view_as(pred)).sum().item()
    print(correct / len(X))