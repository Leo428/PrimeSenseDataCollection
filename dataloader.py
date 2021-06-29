
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_xyz_pairs = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_xyz_pairs)

    def __getitem__(self, idx):
        xyz = self.img_xyz_pairs.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, self.img_xyz_pairs.iloc[idx, 0])
        image = read_image(img_path)
        xyz = self.img_xyz_pairs.iloc[idx, 1]
        f = lambda x : x.strip('][').split(' ')
        xyz = [float(i) for i in f(xyz) if i][:3]
        return image, xyz

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch

dataset = ImageDataset(
    '/home/abhigupta/dev/random/PrimeSenseData/2021-06-29_12-42-49/data.csv',
    '/home/abhigupta/dev/random/PrimeSenseData/2021-06-29_12-42-49/'
    )
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, xyz = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(xyz)
    plt.axis("off")
    img = torch.swapaxes(img, 0, 2)
    img = torch.swapaxes(img, 0, 1)
    plt.imshow(img)
plt.show()
