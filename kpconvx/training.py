import dexc
dexc.install()

import math
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .model import Model
from .subsampling import grid_subsample_fast
from torchvision import transforms

from .ply import read_ply, write_ply


class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


class RandomMirror:
    def __call__(self, pointcloud: np.ndarray):
        return pointcloud * np.array([
            random.choice([1, -1]),
            random.choice([1, -1]),
            1,
        ])


class Subsample:
    def __init__(self, cell_size):
        self.cell_size = cell_size

    def __call__(self, pointcloud):
        return grid_subsample_fast(pointcloud, cell_size=self.cell_size)

class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud).float()


def default_transforms():
    return transforms.Compose([Subsample(0.05), RandomMirror(), RandomRotation_z(), RandomNoise(), ToTensor()])

def test_transforms():
    return transforms.Compose([Subsample(0.05), ToTensor()])


class PointCloudData_RAM(Dataset):
  def __init__(self, root_dir, folder="train", transform=default_transforms()):
      self.root_dir = root_dir
      folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
      self.classes = {folder: i for i, folder in enumerate(folders)}
      self.transforms = transform
      self.data = []
      for category in self.classes.keys():
          new_dir = root_dir+"/"+category+"/"+folder
          for file in os.listdir(new_dir):
              if file.endswith('.ply'):
                  ply_path = new_dir+"/"+file
                  data = read_ply(ply_path)
                  sample = {}
                  sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                  sample['category'] = self.classes[category]
                  self.data.append(sample)

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      pointcloud = self.transforms(self.data[idx]['pointcloud'])
      return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}


train_ds = PointCloudData_RAM('/Users/simon/Developer/iasd/npm/TP6/data/ModelNet10_PLY', folder='train', transform=default_transforms())

loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)


model = Model(10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for _ in range(10):
  model.train()
  optimizer.zero_grad()

  loss_list = []
  correct_count = 0
  n = len(loader)

  for i, x in enumerate(loader):
    cl = x['category'][0]
    points = x['pointcloud'][0]

    logits = model(points)

    loss_list.append(criterion(logits, cl))

    if torch.argmax(logits) == cl:
      correct_count += 1

    if i == n:
      break

  losses = torch.tensor(loss_list, requires_grad=True)
  loss = losses.mean()
  loss.backward()
  optimizer.step()

  print(f'Accuracy: {correct_count / n}')
  print(f'Loss: {loss.item()}')
  print()

  # loss = loss / len(train_ds)
  # print(loss)

  # print(points.min(axis=0).values, points.max(axis=0).values)

  # points_subsampled = grid_subsample_fast(points.numpy(), cell_size=0.05)

  # write_ply('test_default.ply', [points.numpy()], ['x', 'y', 'z'])
  # write_ply('test_subsampled.ply', [points_subsampled], ['x', 'y', 'z'])
  # break

# print(train_ds)
