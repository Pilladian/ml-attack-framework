# Python 3.8.5

from Target import CNNCifar10
from attrinf import create_csv
import torchvision.transforms as transforms
from datasets import UTKFace
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

transform = transforms.Compose([transforms.Resize(size=32),
                              transforms.ToTensor(), 
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset_name = "UTKFace"
dir_path = f"datasets/{dataset_name}"
attr = 'gender'

csv_file = create_csv(dir_path, f"./attrinf-utkface-{attr}.csv", attr)

train_csv_file = 'attrinf-utkface-gender-train.csv'
train_dataset = UTKFace(dir_path, train_csv_file, transform=transform)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1, num_workers=8)

eval_csv_file = 'attrinf-utkface-gender-eval.csv'
eval_dataset = UTKFace(dir_path, eval_csv_file, transform=transform)
eval_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1, num_workers=8)

model = CNNCifar10().to('cpu')

device = 'cuda'
model = CNNCifar10().to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 50
#step_count = len(train_loader_cifar10_target)

model.train()
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.type(torch.int64).to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
        if idx > 500:
            break

    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print ('Training completed...!')
torch.save(model.state_dict(), './CNNCifar10.pth')

num_correct = 0
num_samples = 0
model.eval()

model.eval()
with torch.no_grad():
    for x, y in eval_loader:
        x = x.to(device=device)
        y = y.to(device=device)

        logits = model(x)
        _, preds = torch.max(logits, dim=1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

print(float(num_correct) / float(num_samples))