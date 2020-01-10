import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from torchvision import models

# check for gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(">> device check: ", device)

print(">> preparing data...")
# define a transformation pipeline for our raw data
transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download raw data and then apply transformation pipeline
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# then construct a generator from the transformed data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

# fetch resnet 18
print(">> building model")
num_classes = 10  # cifar10 has 10 classes
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)    # send it to gpu

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9)

n_epochs = 5

print(">> training loop")
for epoch in range(n_epochs):

    start = time.time()
    
    # train phase
    model_ft.train()
    for batch_idx, data in enumerate(trainloader):

        X = data[0].to(device) 
        y = data[1].to(device)

        optimizer_ft.zero_grad()  # clean up gradients
        outputs = model_ft(X)  # emit class logits
        loss = criterion(outputs, y)  # compute loss
        loss.backward()  # compute gradients wrt loss
        optimizer_ft.step()  # update weights according to computed gradients

        # print statistics
        if batch_idx % 10 == 0:
            print("epoch: %d loss: %.3f" % (epoch, loss.item()))


    # validate phase
    model_ft.eval()
    running_corrects = 0
    for X, y in testloader:
        X = X.to(device)
        y = y.to(device)
        output = model_ft(X)
        preds = output.argmax(dim=1, keepdim=False)  # get predictions from argmax
        running_corrects += torch.sum(preds == y)
    epoch_acc = running_corrects.double() / len(testset)
    print("accuracy", epoch_acc.item())

    time_elapsed = time.time() - start
    print("epoch run time", time_elapsed)