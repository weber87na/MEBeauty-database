import os
import pandas as pd
import numpy as np
from progiter import ProgIter
import torch
from torch.autograd import Variable
import cv2
import time
from torchvision import transforms, models, io
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


IMAGE_SIZE = (256, 256)
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]


def read_image_rgb(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    image_data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image file: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class MEBeauty(Dataset):
    
    """Facial Beauty Dataset"""

    def __init__(self, root_dir, train_scores, test_scores, train = True, 
                    transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
       
    
        if train == True:
            folder = train_scores
        else:
            folder = test_scores
            
        self.root_dir = root_dir
        self.images_scores = pd.read_csv(os.path.join(self.root_dir, folder))
        self.transform = transform

    def __len__(self):
        
        return len(self.images_scores)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.images_scores.iloc[idx, 0])
        image = read_image_rgb(img_name)
        score = self.images_scores.iloc[idx, 1]
        if self.transform is not None:
            image = self.transform(image)
            
        return image, score


def build_transform(train_augmentation=False):
    if train_augmentation:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def train_test_data(train_scores, test_scores,root_dir = '', train_augmentation = False, batch = 16, num_workers = 16, pin_memory = True):
    
    # get train and test dataloader from MEBeauty dataset
    
    transform_train = build_transform(train_augmentation)
    transform_test = build_transform(False)
    
    train_data = MEBeauty(root_dir, train_scores,test_scores, train = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch, shuffle =True, num_workers = num_workers, pin_memory = pin_memory)
    test_data = MEBeauty(root_dir, train_scores,test_scores, train = False, transform = transform_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batch, shuffle =False, num_workers = num_workers, pin_memory = pin_memory)
    
    return trainloader, testloader
  
def plot_training(train_losses, valid_losses):
    
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)
    plt.xlabel("epoch")
    plt.plot(train_losses, label="train_loss")
    plt.plot(valid_losses, label="valid_loss")
    plt.legend()
   
