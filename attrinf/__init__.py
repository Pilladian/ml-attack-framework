# Python 3.8.5

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import random
import pandas
import os
from datasets import UTKFace


def create_csv(root, store, attr):
    
    data = pandas.DataFrame(columns=["img_file", attr])
    gender_counter = {0: 0, 1: 0}
    age_counter = dict()
    for a in range(117):
        age_counter[a] = 0
    race_counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    img_files = []
    gender = []
    ages = []
    races = []

    idx = 0
    for i in os.listdir(root):
        l = i.split('_')

        if attr == "gender":
            # female
            if l[1] == '1':
                if gender_counter[int(l[1])] < 700:
                    img_files.append(i)
                    gender.append(int(l[1]))
                    gender_counter[int(l[1])] += 1
                    idx += 1
            # male
            elif l[1] == '0':
                if gender_counter[int(l[1])] < 700:
                    img_files.append(i)
                    gender.append(int(l[1]))
                    gender_counter[int(l[1])] += 1
                    idx += 1

        elif attr == "age":
            if age_counter[int(l[0])] < 5:
                img_files.append(i)
                ages.append(int(l[0]))
                age_counter[int(l[0])] += 1
                idx += 1

        elif attr == "race":
            if race_counter[int(l[2])] < 1692:
                img_files.append(i)
                races.append(int(l[2]))
                race_counter[int(l[2])] += 1
                idx += 1

    data["img_file"] = img_files
    data[attr] = gender if attr == "gender" else ages if attr == "age" else races

    data.to_csv(store, index=False, header=True)
    return store

def get_num_classes(dataset, attr):
    if dataset == "utkface":
            if attr == "race":
                return 5
            elif attr == "age":
                return 117
            elif attr == "gender":
                return 2

def sample_utkface(attr):
    dir_path = "datasets/UTKFace/"
    csv_file = create_csv(dir_path, f"./attrinf-utkface-{attr}.csv", attr)

    transform = transforms.Compose( 
                            [ transforms.Resize(size=32),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

    dataset = UTKFace(dir_path, csv_file, transform=transform)
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=32, num_workers=8)

    return loader

def sample_attack_dataset(dataset, attr):
    if dataset == "utkface":
        return sample_utkface(attr)


class MLP(nn.Module):

    def __init__(self,
                 n_features,
                 n_hidden,
                 n_classes,
                 activation):

        super(MLP, self).__init__()
        self.activation = activation

        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, int(n_hidden/2))
        self.lin3 = nn.Linear(int(n_hidden/2), n_classes)
        self.logso = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        h = self.lin1(inputs)
        h = self.activation(h)
        h = self.lin2(h)
        h = self.activation(h)
        h = self.lin3(h)
        out = self.logso(h)
        return out


class AttributeInferenceAttack:

    def __init__(self, target, device, params):
        self.target = target.to(device)
        self.device = device
        self.params = params
        self.loss_fn = params['loss_fn']
        
    def process_raw_data(self, loader):
        train_data = []
        eval_data = []
        test_data = []
        self.target.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                logits = self.target.get_last_hidden_layer(x)

                for i, logs in enumerate(logits):
                    idx = random.randint(0, 100)
                    d = (list(logs.cpu().numpy()), int(y[i]))
                    if idx < 70:
                        train_data.append(d)
                    elif idx < 80:
                        eval_data.append(d)
                    else:
                        test_data.append(d)
        
        self.train_dl = DataLoader(self.get_data(train_data), batch_size=self.params['batch_size'])
        self.eval_dl = DataLoader(self.get_data(eval_data), batch_size=self.params['batch_size'])
        self.test_dl = DataLoader(self.get_data(test_data), batch_size=self.params['batch_size'])

    def get_data(self, list):
        data = []
        for input, label in list:
            input = torch.FloatTensor(input)
            data.append([input, label])
        return data

    def train_model(self):
        for epoch in range(self.params['epochs']):
            self.model.train()

            for inputs, labels in self.train_dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)

                # backward + optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"\t\t\t\t[1.3.1.1] Epoch [{epoch+1}/{self.params['epochs']}] Loss: {loss.item():0.4f} Acc: {self.evaluate(self.eval_dl):0.4f}", end='\r')
        print()

    def evaluate(self, data):
        num_correct = 0
        num_samples = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in data:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                logits = self.model(x)
                _, predictions = torch.max(logits, dim=1)

                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

        return float(num_correct) / float(num_samples)

    def run(self):
        self.model = MLP(self.params['feat_amount'],       # feature amount
                         self.params['num_hnodes'],        # hidden nodes
                         self.params['num_classes'],       # num classes
                         self.params['activation_fn'])     # activation function

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

        print("\t\t\t[1.3.1] Train Attack Model")
        self.train_model()
        print("\t\t\t[1.3.1] Run Attack against Target model\n\n")
        return self.evaluate(self.test_dl)