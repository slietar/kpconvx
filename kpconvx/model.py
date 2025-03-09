from pathlib import Path

import numpy as np
import torch
from torch import nn

from .kernel import create_kernels
from .kpconv import KPConv
from .ply import write_ply
from .subsampling import grid_subsample, grid_subsample_fast


class Model(nn.Module):
  def __init__(self):
    super().__init__()

    kernel = create_kernels(
      dimension=3,
      kernel_count=1,
      radius=(2.1 * 0.05),
      shell_point_counts=[1, 8, 16],
    )[0, :, :]

    self.kpconv1 = KPConv(
      input_feature_count=1,
      kernel=kernel,
      neighbor_count=10,
      output_feature_count=64,
    )

    self.kpconv2 = KPConv(
      input_feature_count=64,
      kernel=kernel,
      neighbor_count=10,
      output_feature_count=64,
    )

  def forward(self, points1: torch.Tensor):
    features2 = self.kpconv1(points1, points1, torch.ones((points1.size(0), 1)))

    points3, features3 = grid_max_pool(points1, features2, cell_size=0.05 * 2)

    print(points1.size())
    print(points3.size())

    return points3, features3

    # y = self.kpconv2(points, points, x)
    # return y


def grid_max_pool(points: torch.Tensor, features: torch.Tensor, *, cell_size: torch.Tensor | float):
  point_grid_ids = torch.floor(points / cell_size).int()
  unique_voxel_ids, point_voxel_ids, voxel_point_counts = torch.unique(point_grid_ids, dim=0, return_counts=True, return_inverse=True)

  features_subsampled = torch.zeros((len(unique_voxel_ids), features.size(1)), dtype=points.dtype)
  points_subsampled = torch.empty((len(unique_voxel_ids), points.size(1)), dtype=points.dtype)

  for point_index, point_voxel_id in enumerate(point_voxel_ids):
    features_subsampled[point_voxel_id, :] = torch.maximum(features_subsampled[point_voxel_id, :], features[point_index, :])
    points_subsampled[point_voxel_id, :] += points[point_index, :]

  points_subsampled /= voxel_point_counts[:, None]
  return points_subsampled, features_subsampled


class GridKpConvPool(nn.Module):
  def __init__(self, *, cell_size: torch.Tensor | float, feature_count: int, kernel: torch.Tensor):
    super().__init__()

    self.cell_size = cell_size
    self.kpconv = KPConv(
      input_feature_count=feature_count,
      kernel=kernel,
      neighbor_count=10,
      output_feature_count=feature_count,
    )

  def forward(self, points: torch.Tensor, features: torch.Tensor):
    subsampled_points = grid_subsample_fast(points.numpy(), cell_size=self.cell_size)
    return subsampled_points, self.kpconv(subsampled_points, points, features)



kernel = create_kernels(
  dimension=3,
  kernel_count=1,
  radius=(2.1 * 0.05),
  shell_point_counts=[1, 16, 32],
)[0, :, :]


path = Path('s3dis/Area_1/conferenceRoom_1')
points = np.load(path / 'coord.npy') # dtype: float32, shape: (n, 3)
# colors = np.load(path / 'color.npy') # dtype: uint8, shape: (n, 3)
# labels = np.load(path / 'segment.npy') # dtype: int16, shape: (n, 1)

write_ply('test_kernel.ply', [kernel.numpy() + points.min(axis=0)], ['x', 'y', 'z'])

print('Subsampling')
points_subsampled = grid_subsample_fast(points, cell_size=0.05)
print('Model')

# print(points.shape)
# print(points_subsampled.shape)

# write_ply('test_default.ply', [points], ['x', 'y', 'z'])
# write_ply('test_subsampled.ply', [points_subsampled], ['x', 'y', 'z'])

model = Model()
model_points, model_features = model(torch.tensor(points_subsampled))

model_features -= model_features.min()
model_features /= model_features.max()
model_features *= 255
model_features = model_features.int()
# print(conv)

write_ply('test_conv.ply', [model_points.detach().numpy(), model_features[:, :3].detach().numpy()], ['x', 'y', 'z', 'red', 'green', 'blue'])
