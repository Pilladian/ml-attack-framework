# Python 3.8.5
import torchvision.transforms as transforms
import torchvision
import torch.utils.data.dataset as dataset
import torch
import torch.nn as nn
import numpy as np
from datasets import UTKFace
import pandas
from torch.utils.data import DataLoader
import os


# Binary classifier attack model

class BCNet (nn.Module):
    def __init__(self, input_shape):
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train_attack_model(model, train_loader, device):
    learning_rate = 0.01
    num_epochs = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1,1)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # print ('Training completed...!')
    # torch.save(model_attack.state_dict(), './CNNCifar10Attack.pth')
    return model


def sample_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset_cifar10 = torchvision.datasets.CIFAR10(root='datasets/cifar10', train=True, download=False, transform=transform)
    test_dataset_cifar10 = torchvision.datasets.CIFAR10(root='datasets/cifar10', train=False, download=False, transform=transform)

    train_data_cifar10_target, train_data_cifar10_shadow = dataset.random_split(train_dataset_cifar10, [25000, 25000])
    test_data_cifar10_target, test_data_cifar10_shadow = dataset.random_split(test_dataset_cifar10, [5000, 5000])

    train_loader_cifar10_target = torch.utils.data.DataLoader(train_data_cifar10_target, batch_size=32, shuffle=True)
    train_loader_cifar10_shadow = torch.utils.data.DataLoader(train_data_cifar10_shadow, batch_size=32, shuffle=True)
    test_loader_cifar10_target = torch.utils.data.DataLoader(test_data_cifar10_target, batch_size=32, shuffle=False)
    test_loader_cifar10_shadow = torch.utils.data.DataLoader(test_data_cifar10_shadow, batch_size=32, shuffle=False)
    
    return train_loader_cifar10_shadow, test_loader_cifar10_shadow, train_loader_cifar10_target, test_loader_cifar10_target

def get_data(dataset):
    if dataset == 'cifar10':
        return sample_cifar10()

def train_shadow_model(model, device, train_loader):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    num_epochs = 50

    for epoch in range(num_epochs):
        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         
    #print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # print ('Training completed...!')
    # torch.save(model_shadow.state_dict(), './CNNCifar10Shadow.pth')
    return model

def get_attack_train_data(model, train_loader, test_loader, device):
    train_data_attack = []
    train_label_attack = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            for output in outputs.cpu().detach().numpy():
                train_data_attack.append(output)
                train_label_attack.append(1)
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            for output in outputs.cpu().detach().numpy():
                train_data_attack.append(output)
                train_label_attack.append(0)



    train_data = np.array(train_data_attack)
    train_label = np.array(train_label_attack)

    return train_data, train_label


def get_attack_test_data(target, target_train_loader, target_test_loader, device):
    test_data_attack = []
    test_label_attack = []

    with torch.no_grad():
        for images, labels in target_train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = target(images)
            for output in outputs.cpu().detach().numpy():
                test_data_attack.append(output)
                test_label_attack.append(1)
        
        for images, labels in target_test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = target(images)
            for output in outputs.cpu().detach().numpy():
                test_data_attack.append(output)
                test_label_attack.append(0)

    test_data = np.array(test_data_attack)
    test_label = np.array(test_label_attack)

    return test_data, test_label