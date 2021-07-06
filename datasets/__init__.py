# Python 3.8.5


from torch.utils.data import Dataset
import pandas
import os
from PIL import Image
import torch


class UTKFace(Dataset):

    def __init__(self, root, csv_file, train=False, eval=False, test=False, transform=None):
        self.root = root
        self.data_loc = f"{self.root}{'train/' if train else 'eval/' if eval else 'test/' if test else ''}"
        self.data = pandas.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_loc = self.data.iloc[idx, 0]
        image = Image.open(os.path.join(self.data_loc, img_loc)).convert("RGB")
        label = torch.tensor(float(self.data.iloc[idx, 1]))

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