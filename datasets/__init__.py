# Python 3.8.5

from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class UTKFace(Dataset):

    def __init__(self, root, train=False, eval=False, test=False, transform=None):
        self.root = root
        self.data_loc = f"{self.root}{'train/' if train else 'eval/' if eval else 'test/' if test else ''}"
        self.transform = transform
        self.data = self.get_data()
        
    def get_data(self):
        d = {"image_file": [], "label": []}

        for i in os.listdir(self.data_loc):
            l = i.split('_')
            d["image_file"].append(i)
            d["label"].append(l[1]) # gender

        return d

    def __len__(self):
        return len(self.data['image_file'])

    def __getitem__(self, idx):
        img_loc = self.data["image_file"][idx]
        image = Image.open(os.path.join(self.data_loc, img_loc)).convert("RGB")
        label = torch.tensor(float(self.data["label"][idx]))

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)


class CIFAR10(Dataset):

    def __init__(self, root, train=False, eval=False, test=False, transform=None):
        self.root = root
        self.data_loc = f"{self.root}{'train/' if train else 'eval/' if eval else 'test/' if test else ''}"
        self.transform = transform
        self.data = self.get_data()
        
    def get_data(self):
        d = {"image_file": [], "label": []}

        for i in os.listdir(self.data_loc):
            l = i.split('_')
            d["image_file"].append(i)
            d["label"].append(l[0]) # airplane, ship, ...

        return d

    def __len__(self):
        return len(self.data['image_file'])

    def __getitem__(self, idx):
        img_loc = self.data["image_file"][idx]
        image = Image.open(os.path.join(self.data_loc, img_loc)).convert("RGB")
        label = torch.tensor(float(self.data["label"][idx]))

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)


class MIADataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length