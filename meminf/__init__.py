# Python 3.8.5
import torchvision.transforms as transforms
import torchvision
import torch.utils.data.dataset as dataset
import torch
import torch.nn as nn
import numpy as np


from datasets import CIFAR10, UTKFace, MNIST, ATT


# Attack Model
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


def sample_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # datasets/CIFAR10/train was used to train the tagret model -> Use it to sample Attack Dataset pos. samples
    train_data_cifar10_target = CIFAR10('datasets/CIFAR10/', train=True, transform=transform)
    # datasets/CIFAR10/eval was not used to train the tagret model -> Use it to sample Attack Dataset neg. samples
    test_data_cifar10_target = CIFAR10('datasets/CIFAR10/', eval=True, transform=transform)
    
    # Shadow Model training dataset from the same domain as the training dataset of the target model
    shadow = CIFAR10('datasets/CIFAR10/', test=True, transform=transform)
    length = shadow.__len__()
    l = int(shadow.__len__() / 2)
    train_data_cifar10_shadow, test_data_cifar10_shadow = dataset.random_split(shadow, [l, length - l])

    train_loader_cifar10_target = torch.utils.data.DataLoader(train_data_cifar10_target, batch_size=32, shuffle=True)
    test_loader_cifar10_target = torch.utils.data.DataLoader(test_data_cifar10_target, batch_size=32, shuffle=False)
    train_loader_cifar10_shadow = torch.utils.data.DataLoader(train_data_cifar10_shadow, batch_size=32, shuffle=True)
    test_loader_cifar10_shadow = torch.utils.data.DataLoader(test_data_cifar10_shadow, batch_size=32, shuffle=False)
    
    return train_loader_cifar10_shadow, test_loader_cifar10_shadow, train_loader_cifar10_target, test_loader_cifar10_target

def sample_utkface():
    transform = transforms.Compose([transforms.Resize(size=32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # datasets/UTKFace/train was used to train the tagret model -> Use it to sample Attack Dataset pos. samples
    train_data_utkface_target = UTKFace('datasets/UTKFace/', train=True, transform=transform)
    # datasets/UTKFace/eval was not used to train the tagret model -> Use it to sample Attack Dataset neg. samples
    test_data_utkface_target = UTKFace('datasets/UTKFace/', eval=True, transform=transform)
    
    # Shadow Model training dataset from the same domain as the training dataset of the target model
    shadow = UTKFace('datasets/UTKFace/', test=True, transform=transform)
    length = shadow.__len__()
    l = int(shadow.__len__() / 2)
    train_data_utkface_shadow, test_data_utkface_shadow = dataset.random_split(shadow, [l, length - l])

    train_loader_utkface_target = torch.utils.data.DataLoader(train_data_utkface_target, batch_size=32, shuffle=True)
    test_loader_utkface_target = torch.utils.data.DataLoader(test_data_utkface_target, batch_size=32, shuffle=False)
    train_loader_utkface_shadow = torch.utils.data.DataLoader(train_data_utkface_shadow, batch_size=64, shuffle=True)
    test_loader_utkface_shadow = torch.utils.data.DataLoader(test_data_utkface_shadow, batch_size=64, shuffle=False)
    
    return train_loader_utkface_shadow, test_loader_utkface_shadow, train_loader_utkface_target, test_loader_utkface_target

def sample_mnist():
    transform = transforms.Compose([transforms.Resize(size=32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # datasets/MNIST/train was used to train the tagret model -> Use it to sample Attack Dataset pos. samples
    train_data_mnist_target = MNIST('datasets/MNIST/', train=True, transform=transform)
    # datasets/MNIST/eval was not used to train the tagret model -> Use it to sample Attack Dataset neg. samples
    test_data_mnist_target = MNIST('datasets/MNIST/', eval=True, transform=transform)
    
    # Shadow Model training dataset from the same domain as the training dataset of the target model
    shadow = MNIST('datasets/MNIST/', test=True, transform=transform)
    length = shadow.__len__()
    l = int(shadow.__len__() / 2)
    train_data_mnist_shadow, test_data_mnist_shadow = dataset.random_split(shadow, [l, length - l])

    train_loader_mnist_target = torch.utils.data.DataLoader(train_data_mnist_target, batch_size=32, shuffle=True)
    test_loader_mnist_target = torch.utils.data.DataLoader(test_data_mnist_target, batch_size=32, shuffle=False)
    train_loader_mnist_shadow = torch.utils.data.DataLoader(train_data_mnist_shadow, batch_size=64, shuffle=True)
    test_loader_mnist_shadow = torch.utils.data.DataLoader(test_data_mnist_shadow, batch_size=64, shuffle=False)
    
    return train_loader_mnist_shadow, test_loader_mnist_shadow, train_loader_mnist_target, test_loader_mnist_target

def sample_att():
    transform = transforms.Compose([transforms.Resize(size=(32,32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # datasets/MNIST/train was used to train the tagret model -> Use it to sample Attack Dataset pos. samples
    train_data_mnist_target = ATT('datasets/ATT/', train=True, transform=transform)
    # datasets/MNIST/eval was not used to train the tagret model -> Use it to sample Attack Dataset neg. samples
    test_data_mnist_target = ATT('datasets/ATT/', eval=True, transform=transform)
    
    # Shadow Model training dataset from the same domain as the training dataset of the target model
    shadow = ATT('datasets/ATT/', test=True, transform=transform)
    length = shadow.__len__()
    l = int(shadow.__len__() / 2)
    train_data_mnist_shadow, test_data_mnist_shadow = dataset.random_split(shadow, [l, length - l])

    train_loader_mnist_target = torch.utils.data.DataLoader(train_data_mnist_target, batch_size=32, shuffle=True)
    test_loader_mnist_target = torch.utils.data.DataLoader(test_data_mnist_target, batch_size=32, shuffle=False)
    train_loader_mnist_shadow = torch.utils.data.DataLoader(train_data_mnist_shadow, batch_size=64, shuffle=True)
    test_loader_mnist_shadow = torch.utils.data.DataLoader(test_data_mnist_shadow, batch_size=64, shuffle=False)
    
    return train_loader_mnist_shadow, test_loader_mnist_shadow, train_loader_mnist_target, test_loader_mnist_target

def get_data(dataset):
    if dataset == 'cifar10':
        return sample_cifar10()
    elif dataset == 'utkface':
        return sample_utkface()
    elif dataset == 'mnist':
        return sample_mnist()
    elif dataset == 'att':
        return sample_att()

def train_shadow_model(model, device, train_loader, ):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 50

    for epoch in range(num_epochs):
        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.type(torch.int64).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         
        print(f"\t     [2.3] Create and Train Shadow model: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}", end='\r')
    print()

    return model

def train_attack_model(model, train_loader, device):
    learning_rate = 0.001
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            
        print(f"\t     [2.7] Train Attack Model: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}", end='\r')
    print()

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
                output.sort()
                output = output[-3:]
                train_data_attack.append(output)
                train_label_attack.append(1)
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            for output in outputs.cpu().detach().numpy():
                output.sort()
                output = output[-3:]
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
                output.sort()
                output = output[-3:]
                test_data_attack.append(output)
                test_label_attack.append(1)

        for images, labels in target_test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = target(images)
            for output in outputs.cpu().detach().numpy():
                output.sort()
                output = output[-3:]
                test_data_attack.append(output)
                test_label_attack.append(0)

    test_data = np.array(test_data_attack)
    test_label = np.array(test_label_attack)

    return test_data, test_label

def eval_attack_model(model_attack, test_loader, device):
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1).detach().cpu().numpy()
            outputs = model_attack(data)
            outputs = outputs.reshape(-1).detach().cpu().numpy().round()
            
            for i in range(min(32, data.size(0))):
                label = labels[i]
                pred = outputs[i]
                if (label == pred and label != 0):
                    tp += 1
                elif (label != pred and label != 0):
                    fn += 1
                elif (label != pred and label == 0):
                    fp += 1
                if (label == pred):
                    n_correct += 1
                n_samples += 1

        acc = float(n_correct / n_samples)
    
    return acc
