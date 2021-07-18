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


class MNIST(Dataset):

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

class ATT(Dataset):

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
            d["label"].append(l[0]) # subject nr.

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

class AttributeInferenceAttackRawDataset(Dataset):

    def __init__(self, root, attr, transform=None):
        self.root = root
        self.attr = attr
        self.transform = transform
        self.data = self.get_data()
        
    def get_data(self):
        d = {"image_file": [], "label": []}

        if 'UTKFace' in self.root:
            lookup = {'age': 0, 'gender': 1, 'race': 2}
        elif 'ATT' in self.root:
            lookup = {'no-glasses': 0, 'glasses': 1}

        # collect all images from train eval and test directory
        for i in os.listdir(self.root + 'train/'):
            l = i.split('_')
            d["image_file"].append('train/' + i)
            d["label"].append(l[lookup[self.attr]])

        for i in os.listdir(self.root + 'eval/'):
            l = i.split('_')
            d["image_file"].append('eval/' + i)
            d["label"].append(l[lookup[self.attr]]) 

        for i in os.listdir(self.root + 'test/'):
            l = i.split('_')
            d["image_file"].append('test/' + i)
            d["label"].append(l[lookup[self.attr]])

        # sample same amount of every attribute category
        final_data = {"image_file": [], "label": []}

        if 'UTKFace' in self.root:
            utkface_counter = {'age': {}, 'gender': {}, 'race': {}}

            for idx in range(len(d['image_file'])):
                image = d['image_file'][idx]
                label = d['label'][idx]

                # age
                if self.attr == 'age':
                    if label not in utkface_counter['age']:
                        utkface_counter['age'][label] = 0

                    elif utkface_counter['age'][label] < 5:
                        utkface_counter['age'][label] += 1

                    else:
                        continue

                    final_data["image_file"].append(image)
                    final_data["label"].append(label)

                # gender
                elif self.attr == 'gender':
                    if label not in utkface_counter['gender']:
                        utkface_counter['gender'][label] = 0

                    elif utkface_counter['gender'][label] < 10000:
                        utkface_counter['gender'][label] += 1

                    else:
                        continue

                    final_data["image_file"].append(image)
                    final_data["label"].append(label)

                # race
                elif self.attr == 'race':
                    if label not in utkface_counter['race']:
                        utkface_counter['race'][label] = 0

                    elif utkface_counter['race'][label] < 1692:
                        utkface_counter['race'][label] += 1

                    else:
                        continue

                    final_data["image_file"].append(image)
                    final_data["label"].append(label)

        if 'ATT' in self.root:
            att_counter = {}

            for idx in range(len(d['image_file'])):
                image = d['image_file'][idx]
                label = d['label'][idx]

                # wears glasses
                if label not in att_counter:
                    att_counter[label] = 0

                elif att_counter[label] < 149:
                    att_counter[label] += 1

                else:
                    continue

                final_data["image_file"].append(image)
                final_data["label"].append(label)

        return final_data

    def __len__(self):
        return len(self.data['image_file'])

    def __getitem__(self, idx):
        img_loc = self.data["image_file"][idx]
        image = Image.open(os.path.join(self.root, img_loc)).convert("RGB")
        label = torch.tensor(float(self.data["label"][idx]))

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)


class AttributeInferenceAttackDataset(Dataset):

    def __init__(self, data):
        self.data = self.modify_data(data)

    def modify_data(self, data):
        modified_data = []
        for post, label in data:
            post = torch.FloatTensor(post)
            modified_data.append([post, label])
        return modified_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        post = self.data[idx][0]
        label = self.data[idx][1]
        return (post, label)


class MembershipInferenceAttackDataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length