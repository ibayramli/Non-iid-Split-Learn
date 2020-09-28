from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.datasets import MNIST, FashionMNIST
from os.path import join
from PIL import Image

import torchvision.transforms as tt
import torch
import os
import json

with open('./src/utils/cifar100_class_maps.txt', 'r') as f:
    CLASS_MAP_CIFAR_100 = json.load(f)
with open('./src/utils/cifar10_class_maps.txt', 'r') as f:
    CLASS_MAP_CIFAR_10 = json.load(f)

    
class CifarDataLoader(Dataset):
    def __init__(self, src, class_map, transforms=None):
        """
        src: path to source directory organized as follows:
            src
              |_dogs
                  |_ 1.png
                  |_ 2.png
                  ...
              |_cats
                  |_ 11.png
                  ...
        
        class_map: mapping of classes to indices. A class map is needed to allow for general labeling of classes
                   even when the dataloader is called on a client path with partial labels. For example, label 5 
                   for CIFAR10 data will denote a cat for all classes even if 6th (0-based index system) folder  
                   of client 1 is dog samples. Class maps also allow generalization to both CIFAR10 / 100 data.
        """
        super(Dataset, self).__init__()
        self.class_map = class_map
        self.img_paths, self.labels = self._get_cifar_data(src)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.img_paths)
                
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        
        return (img, label)
    
    def _get_cifar_data(self, src):
        img_paths = []
        labels = []
        
        classes = os.listdir(src)
        for c in classes:
            c_names = os.listdir(os.path.join(src, c))
            c_paths = [os.path.join(src, c, name) for name in c_names]
            c_int = self._class_to_int(c)
            
            img_paths.extend(c_paths)
            labels.extend([c_int] * len(c_paths))
        
        return img_paths, labels
            
    def _class_to_int(self, c):
        return self.class_map[c]


def cifar_100_dataloader(root, train=True, **kwargs):
    """
    root: folder where ./train and ./test files are located in
    
    train: if True access data from ./train, else use ./test
    
    """
    transforms = tt.Compose([
        tt.RandomHorizontalFlip(),
        tt.RandomCrop(32, padding=4),
        tt.ToTensor(),
        tt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]) 
    
    path = join(root, 'train') if train else join(root, 'test')
        
    dataset = CifarDataLoader(path, CLASS_MAP_CIFAR_100, transforms)
    num_samples = int(len(dataset) * 1.5)
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    dataloader = DataLoader(dataset, **kwargs, sampler=sampler)
    
    return dataloader
    

def cifar_10_dataloader(root, train=True, **kwargs):
    """
    root: folder where ./train and ./test files are located in
    
    train: if True access data from ./train, else use ./test
    
    """
    transforms = tt.Compose([
        tt.RandomHorizontalFlip(),
        tt.RandomCrop(32, padding=4),
        tt.ToTensor(),
        tt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]) 
    
    path = join(root, 'train') if train else join(root, 'test')
        
    dataset = CifarDataLoader(path, CLASS_MAP_CIFAR_10, transforms)
    num_samples = int(len(dataset) * 1.5)
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    dataloader = DataLoader(dataset, **kwargs, sampler=sampler)
    
    return dataloader


def MNIST_dataloader(root, train, shuffle=True, **kwargs):
    """
    root: folder where ./train and ./test files are located in
    
    train: if True access data from ./train, else use ./test
    
    shuffle: shuffle the samples in each epoch
    
    """
    RandAffine = tt.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2))
    transforms = tt.Compose([
        RandAffine,
        tt.ToTensor(),
        tt.Normalize((0.1307,), (0.3081,)) # (0.1307,), (0.3081,) are the mean and std of MNIST training set
    ])
    dataset = MNIST(root, train=train, transform=transforms, 
                    target_transform=None, download=False)
    dataloader = DataLoader(dataset, shuffle=shuffle, **kwargs)
    
    return dataloader


def FashionMNIST_dataloader(root, train, shuffle=True, **kwargs):
    """
    root: folder where ./train and ./test files are located in
    
    train: if True access data from ./train, else use ./test
    
    shuffle: shuffle the samples in each epoch
    
    """
    RandAffine = tt.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2))
    transforms = tt.Compose([
        RandAffine,
        tt.ToTensor(),
        tt.Normalize((0.1307,), (0.3081,))
    ])
    dataset = FashionMNIST(root, train=train, transform=transforms, 
                    target_transform=None, download=False)
    dataloader = DataLoader(dataset, shuffle=shuffle, **kwargs)
    
    return dataloader

