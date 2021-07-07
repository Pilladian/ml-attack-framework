# Python 3.8.5
import torchvision.transforms as transforms
import torchvision
import torch.utils.data.dataset as dataset
import torch
import torch.nn as nn
import numpy as np


from datasets import CIFAR10, UTKFace


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
    train_data_cifar10_shadow, test_data_cifar10_shadow = dataset.random_split(shadow, [6433, 6434])

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
    train_data_utkface_shadow, test_data_utkface_shadow = dataset.random_split(shadow, [2541, 2540])

    train_loader_utkface_target = torch.utils.data.DataLoader(train_data_utkface_target, batch_size=32, shuffle=True)
    test_loader_utkface_target = torch.utils.data.DataLoader(test_data_utkface_target, batch_size=32, shuffle=False)
    train_loader_utkface_shadow = torch.utils.data.DataLoader(train_data_utkface_shadow, batch_size=64, shuffle=True)
    test_loader_utkface_shadow = torch.utils.data.DataLoader(test_data_utkface_shadow, batch_size=64, shuffle=False)
    
    return train_loader_utkface_shadow, test_loader_utkface_shadow, train_loader_utkface_target, test_loader_utkface_target

def get_data(dataset):
    if dataset == 'cifar10':
        return sample_cifar10()
    elif dataset == 'utkface':
        return sample_utkface()

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
            
        print(f"\t     [2.5] Train Attack Model: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}", end='\r')
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



# ----------------------------------------------------------------------------------
# from datasets import AttributeInferenceAttackDataset
# from torch.utils.data import DataLoader

# # Attack Model
# class MLP(nn.Module):

#     def __init__(self, parameter):

#         super(MLP, self).__init__()

#         n_features = parameter['n_input_nodes']
#         n_hidden = parameter['n_hidden_nodes']
#         n_classes = parameter['n_output_nodes']
#         self.activation = parameter['activation_fn']
#         self.lin1 = nn.Linear(n_features, n_hidden)
#         self.lin2 = nn.Linear(n_hidden, int(n_hidden/2))
#         self.lin3 = nn.Linear(int(n_hidden/2), n_classes)
#         self.logsoft = nn.LogSoftmax(dim=1)

#     def forward(self, inputs):
#         h = self.lin1(inputs)
#         h = self.activation(h)
#         h = self.lin2(h)
#         h = self.activation(h)
#         h = self.lin3(h)
#         out = self.logsoft(h)
#         return out


# def train_attack_model(model, train_loader, epochs, loss_fn, optimizer, device):
#     for epoch in range(epochs):
#         model.train()

#         for inputs, labels in train_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             # forward
#             output = model(inputs)
#             loss = loss_fn(output, labels)

#             # backward + optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f"\t\t[2.5] Train Attack Model: Epoch [{epoch+1}/{epochs}] Loss: {loss.item():0.4f}", end='\r')
#     print()

# def get_attack_train_data(model, train_loader, test_loader, device):
#     with torch.no_grad():
#         data = []

#         for images, labels in train_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             for i, logs in enumerate(outputs):
#                 a = list(logs.cpu().numpy())
#                 a.sort()
#                 a = a[-3:]
#                 data.append((a, int(labels[i])))
        
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             for i, logs in enumerate(outputs):
#                 a = list(logs.cpu().numpy())
#                 a.sort()
#                 a = a[-3:]
#                 data.append((a, int(labels[i])))

#     attack_dataset = AttributeInferenceAttackDataset(data)
#     train_attack_loader = DataLoader(attack_dataset, batch_size=32, shuffle=True)

#     return train_attack_loader

# def get_attack_test_data(model, train_loader, test_loader, device):
#     with torch.no_grad():
#         data = []

#         for images, labels in train_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             for i, logs in enumerate(outputs):
#                 a = list(logs.cpu().numpy())
#                 a.sort()
#                 a = a[-3:]
#                 data.append((a, int(labels[i])))
        
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             for i, logs in enumerate(outputs):
#                 a = list(logs.cpu().numpy())
#                 a.sort()
#                 a = a[-3:]
#                 data.append((a, int(labels[i])))

#     attack_dataset = AttributeInferenceAttackDataset(data)
#     test_attack_loader = DataLoader(attack_dataset, batch_size=32, shuffle=True)

#     return test_attack_loader

# def eval_attack_model(model, loader, device):
#     num_correct = 0
#     num_samples = 0

#     model.eval()
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)

#             logits = model(x)
#             _, predictions = torch.max(logits, dim=1)

#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)

#     return float(num_correct) / float(num_samples)