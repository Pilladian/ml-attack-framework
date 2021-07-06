# Python 3.8.5


import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import os
import sys


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 32*32 -> 28*28
        self.pool = nn.MaxPool2d(2, 2) # 28*28 -> 14*14
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(400, 200) # 16*5*5 = 400
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x.view(-1, 400)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_last_hidden_layer(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x.view(-1, 400)))
        x = self.fc2(x)
        return x


# # Convolutional Neural Network trained to infer gender
# class CNNN(nn.Module):

#     def __init__(self):
#         super(CNN, self).__init__()

#         # input_channels, output_channels, kernel_size
#         self.conv1 = nn.Conv2d(3, 32, 3)
#         self.conv2 = nn.Conv2d(32, 64, 3)

#         self.lin1 = nn.Linear(774400, 1024)
#         self.lin2 = nn.Linear(1024, 512)
#         self.lin3 = nn.Linear(512, 256)
#         self.lin4 = nn.Linear(256, 2)

#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         # convolution
#         h = self.conv1(x)
#         h = F.relu(h)
#         h = self.conv2(h)
#         h = F.relu(h)
#         h = F.max_pool2d(h, 2)
#         h = self.dropout(h)
#         h = torch.flatten(h, 1)

#         # classifier
#         h = self.lin1(h)
#         h = F.relu(h)
#         h = self.dropout(h)
#         h = self.lin2(h)
#         h = F.relu(h)
#         h = self.dropout(h)
#         h = self.lin3(h)
#         h = F.relu(h)
#         h = self.dropout(h)
#         h = self.lin4(h)
#         output = F.log_softmax(h, dim=1)
#         return output

#     def get_last_hidden_layer(self, x):
#         # convolution
#         h = self.conv1(x)
#         h = F.relu(h)
#         h = self.conv2(h)
#         h = F.relu(h)
#         h = F.max_pool2d(h, 2)
#         h = self.dropout(h)
#         h = torch.flatten(h, 1)

#         # classifier
#         h = self.lin1(h)
#         h = F.relu(h)
#         h = self.dropout(h)
#         h = self.lin2(h)
#         h = F.relu(h)
#         h = self.dropout(h)
#         output = self.lin3(h)
#         return output


# class Target:

#     def __init__(self, device="cpu", train=False, ds_root=None):
#         self.device = device
#         self.model = CNN().to(self.device)  # TransCNN(1024, 5).to(self.device)
#         self.train = train

#         if self.train:
#             self._train_model(ds_root=ds_root)

#         self._load_model()

#     def _train_model(self, ds_root):
#         print(f' [+] Load dataset')
#         self._load_dataset(ds_root)

#         self.lfn = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
#         val_acc = self.evaluate_model(self.model, self.validation_loader)
#         print(f' [+] Accuracy of untrained model: {val_acc}')

#         eps = 10
#         max_acc = 0
#         print(f' [+] Train CNN on {self.device}')
#         for epoch in range(eps):
#             self.model.train()
#             for imgs, labels in self.train_loader:
#                 imgs = imgs.to(self.device)
#                 labels = labels.type(torch.int64).to(self.device)
#                 outputs = self.model(imgs)
#                 loss = self.lfn(outputs, labels)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#             val_acc = self.evaluate_model(self.model, self.validation_loader)
#             print(f' Epoch {epoch + 1}/{eps} : {val_acc}')
#             if val_acc > max_acc:
#                 max_acc = val_acc
#                 self._save_model()

#     def _load_dataset(self, dir):
#         self.transform = transforms.Compose(
#                             [ transforms.Resize(size=256),
#                               transforms.CenterCrop(size=224),
#                               transforms.ToTensor(),
#                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                             ])

#         train_set = UTKFace.UTKFace('UTKFace', train=True, transform=self.transform)
#         self.train_loader = DataLoader(dataset=train_set,
#                                        shuffle=True,
#                                        batch_size=64,
#                                        num_workers=8)

#         validation_set = UTKFace.UTKFace('UTKFace', eval=True, transform=self.transform)
#         self.validation_loader = DataLoader(dataset=validation_set,
#                                             shuffle=True,
#                                             batch_size=64,
#                                             num_workers=8)

#     def _save_model(self):
#         torch.save(self.model.state_dict(), f'Target/model.pt')

#     def _load_model(self):
#         self.model.load_state_dict(torch.load("Target/attribute-inference-attack-target.pt", map_location=self.device))

#     def evaluate_model(self, model, data):
#         num_correct = 0
#         num_samples = 0
#         model.eval()

#         model.eval()
#         with torch.no_grad():
#             for x, y in data:
#                 x = x.to(device=self.device)
#                 y = y.to(device=self.device)

#                 logits = model(x)
#                 _, preds = torch.max(logits, dim=1)
#                 num_correct += (preds == y).sum()
#                 num_samples += preds.size(0)

#         return float(num_correct) / float(num_samples)
