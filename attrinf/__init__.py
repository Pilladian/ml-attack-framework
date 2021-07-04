# Python 3.8.5

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import random
import pandas
import os
from datasets import UTKFace
import pickle


def create_csvv(root, store, attr, partition):

    data = pandas.DataFrame(columns=["img_file", attr])
    data["img_file"] = os.listdir(f'{root}/{partition}')
    
    for idx, i in enumerate(os.listdir(f'{root}/{partition}')):
        l = i.split('_')
            
        if attr == "gender":
            # female
            if l[1] == '1':
                data[attr][idx] = 1
            # male
            elif l[1] == '0':
                data[attr][idx] = 0

        elif attr == "age":
            data[attr][idx] = int(l[0])

        elif attr == "race":
            data[attr][idx] = int(l[2])

    data.to_csv(store, index=False, header=True)
    return store


def create_csvvvvvv(root, store, attr, partition):
    
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
    for i in os.listdir(f'{root}/{partition}'):
        print(i)
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
            if race_counter[int(l[2])] < 1000:
                img_files.append(i)
                races.append(int(l[2]))
                race_counter[int(l[2])] += 1
                idx += 1

    data["img_file"] = img_files
    data[attr] = gender if attr == "gender" else ages if attr == "age" else races


    data.to_csv(store, index=False, header=True)
    return store


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
            if race_counter[int(l[2])] < 1000:
                img_files.append(i)
                races.append(int(l[2]))
                race_counter[int(l[2])] += 1
                idx += 1

    data["img_file"] = img_files
    data[attr] = gender if attr == "gender" else ages if attr == "age" else races

    data.to_csv(store, index=False, header=True)
    return store


def sample_utkface(attr):
    dir_path = "datasets/UTKFace/"

    csv_file = create_csv(dir_path, f"./attrinf-utkface-{attr}.csv", attr)

    # csv_train = create_csv(dir_path, f"./attrinf-utkface-{attr}-train.csv", attr, 'train')
    # csv_eval = create_csv(dir_path, f"./attrinf-utkface-{attr}-eval.csv", attr, 'eval')
    # csv_test= create_csv(dir_path, f"./attrinf-utkface-{attr}-test.csv", attr, 'test')

    transform = transforms.Compose( 
                            [ transforms.Resize(size=256),
                              transforms.CenterCrop(size=224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

    # train_set = UTKFace(dir_path, csv_train, train=True, transform=transform)
    # train_loader = DataLoader(dataset=train_set,
    #                           shuffle=True,
    #                           batch_size=32,
    #                           num_workers=8)

    # validation_set = UTKFace(dir_path, csv_eval, eval=True, transform=transform)
    # validation_loader = DataLoader(dataset=validation_set,
    #                                shuffle=True,
    #                                batch_size=32,
    #                                num_workers=8)

    dataset = UTKFace(dir_path, csv_file, transform=transform)
    loader = DataLoader(dataset=dataset,
                             shuffle=True,
                             batch_size=32,
                             num_workers=8)

    return loader


def sample_attack_dataset(dataset, attr):
    if dataset == "utkface":
        return sample_utkface(attr)


class MLP(nn.Module):

    def __init__(self,
                 n_features,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout):

        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, int(n_hidden/2))
        self.lin3 = nn.Linear(int(n_hidden/2), n_classes)
        self.logso = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        h = self.lin1(inputs)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.lin2(h)
        h = self.activation(h)
        h = self.dropout(h)
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
                #logits = self.target(x)

                for i, logs in enumerate(logits):
                    idx = random.randint(0, 100)
                    d = (list(logs.numpy()), int(y[i]))
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
                print((predictions == y).sum() / len(logits))
                num_samples += predictions.size(0)

        return float(num_correct) / float(num_samples)

    def run(self):
        self.model = MLP(self.params['feat_amount'],       # feature amount
                         self.params['num_hnodes'],        # hidden nodes
                         self.params['num_classes'],       # num classes
                         self.params['activation_fn'],     # activation function
                         self.params['dropout'])           # dropout

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        self.train_model()

        return self.evaluate(self.test_dl)