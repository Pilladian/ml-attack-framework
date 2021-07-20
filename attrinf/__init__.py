# Python 3.8.5

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data.dataset as dataset
import torch.nn as nn
import torch

from datasets import AttributeInferenceAttackRawDataset, AttributeInferenceAttackDataset


# Attack Model
class MLP(nn.Module):

    def __init__(self, parameter):

        super(MLP, self).__init__()

        n_features = parameter['n_input_nodes']
        n_hidden = parameter['n_hidden_nodes']
        n_classes = parameter['n_output_nodes']
        self.activation = parameter['activation_fn']
        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, int(n_hidden/2))
        self.lin3 = nn.Linear(int(n_hidden/2), n_classes)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        h = self.lin1(inputs)
        h = self.activation(h)
        h = self.lin2(h)
        h = self.activation(h)
        h = self.lin3(h)
        out = self.logsoft(h)
        return out


def get_raw_attack_dataset(dataset, attr):
    transform = transforms.Compose( 
                            [ transforms.Resize(size=(32,32)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
    
    if dataset == 'utkface':
        dir_path = 'datasets/UTKFace/'
    elif dataset == 'att':
        dir_path = 'datasets/ATT/'

    raw_set = AttributeInferenceAttackRawDataset(dir_path, attr=attr, transform=transform)
    raw_loader = torch.utils.data.DataLoader(raw_set, batch_size=32, shuffle=True)

    return raw_loader

def get_attack_loader(target_model, raw_loader, device):
    data = []
    
    target_model.eval()
    with torch.no_grad():
        for x, y in raw_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            logits = target_model.get_last_hidden_layer(x)

            for i, logs in enumerate(logits):
                data.append((list(logs.cpu().numpy()), int(y[i])))

    attack_dataset = AttributeInferenceAttackDataset(data)

    size_train = int(attack_dataset.__len__() / 2)
    size_tmp = attack_dataset.__len__() - size_train
    size_eval = int(size_tmp / 3)
    size_test = size_tmp - size_eval

    train_attack_dataset, tmp_dataset = dataset.random_split(attack_dataset, [size_train, size_tmp])
    eval_attack_dataset, test_attack_dataset = dataset.random_split(tmp_dataset, [size_eval , size_test])
    
    train_attack_loader = DataLoader(train_attack_dataset, batch_size=32, shuffle=True)
    eval_attack_loader = DataLoader(eval_attack_dataset, batch_size=32, shuffle=True)
    test_attack_loader = DataLoader(test_attack_dataset, batch_size=32)

    return train_attack_loader, eval_attack_loader, test_attack_loader
        
def train_attack_model(model, train_loader, epochs, loss_fn, optimizer, device):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            output = model(inputs)
            loss = loss_fn(output, labels)

            # backward + optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"\t     [1.5] Train Attack Model: Epoch [{epoch+1}/{epochs}] Loss: {loss.item():0.4f}", end='\r')
    print()

def eval_attack_model(model, loader, device):
    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            logits = model(x)
            _, predictions = torch.max(logits, dim=1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return float(num_correct) / float(num_samples)
