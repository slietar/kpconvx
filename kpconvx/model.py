import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .kernel import create_kernels
from .kpconv import KPConv
from .ply import write_ply
from .subsampling import grid_subsample, grid_subsample_fast

print(os.getpid())

np.random.seed(0)
torch.manual_seed(0)

class Model(nn.Module):
  def __init__(self):
    super().__init__()

    kernel_path = Path('kernel.npy')

    if kernel_path.exists():
      kernel = torch.tensor(np.load(kernel_path))
    else:
      kernel = create_kernels(
        dimension=3,
        kernel_count=1,
        radius=(2.1 * 0.05),
        shell_point_counts=[1, 8, 16],
      )[0, :, :]

      np.save(kernel_path, kernel.numpy())

    self.kpconv1 = KPConv(
      input_feature_count=1,
      kernel=kernel,
      neighbor_count=10,
      output_feature_count=64,
    )

    self.kpconv2 = KPConv(
      input_feature_count=64,
      kernel=(kernel * 2),
      neighbor_count=10,
      output_feature_count=64,
    )

  def forward(self, points1: torch.Tensor):
    features2 = self.kpconv1(points1, points1, torch.ones((points1.size(0), 1)))

    points3, features3 = grid_max_pool(points1, features2, cell_size=0.05 * 2)

    # print(points1.size())
    # print(points3.size())

    return points3, features3

    # y = self.kpconv2(points, points, x)
    # return y


def grid_max_pool_undifferentiable(points: torch.Tensor, features: torch.Tensor, *, cell_size: torch.Tensor | float):
  point_grid_ids = torch.floor(points / cell_size).int()
  unique_voxel_ids, point_voxel_ids, voxel_point_counts = torch.unique(point_grid_ids, dim=0, return_counts=True, return_inverse=True)

  features_subsampled = torch.full((len(unique_voxel_ids), features.size(1)), -torch.inf, dtype=points.dtype)
  points_subsampled = torch.zeros((len(unique_voxel_ids), points.size(1)), dtype=points.dtype)

  for point_index, point_voxel_id in enumerate(point_voxel_ids):
    features_subsampled[point_voxel_id, :] = torch.maximum(features_subsampled[point_voxel_id, :], features[point_index, :])
    points_subsampled[point_voxel_id, :] += points[point_index, :]

  points_subsampled /= voxel_point_counts[:, None]
  return points_subsampled, features_subsampled


def nanmax(tensor: torch.Tensor, dim=None, keepdim=False):
  min_value = torch.finfo(tensor.dtype).min
  return tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim).values

def grid_max_pool(points: torch.Tensor, features: torch.Tensor, *, cell_size: torch.Tensor | float):
  point_grid_ids = torch.floor(points / cell_size).int()
  unique_voxel_ids, point_voxel_indices, voxel_point_counts = torch.unique(point_grid_ids, dim=0, return_counts=True, return_inverse=True)

  counts = torch.zeros(len(unique_voxel_ids), dtype=torch.int64)
  indices = torch.zeros((len(unique_voxel_ids), voxel_point_counts.max()), dtype=torch.int)
  mask = torch.full((len(unique_voxel_ids), voxel_point_counts.max()), torch.nan)

  for point_index, point_voxel_index in enumerate(point_voxel_indices):
    point_index_in_voxel = counts[point_voxel_index].item()
    indices[point_voxel_index, point_index_in_voxel] = point_index
    mask[point_voxel_index, point_index_in_voxel] = 1.0
    counts[point_voxel_index] += 1

  assert torch.allclose(counts, voxel_point_counts)

  gathered = points[indices, :] * mask[:, :, None]
  points_subsampled = gathered.nanmean(dim=1) # / voxel_point_counts[:, None]

  # print(gathered.size())
  # print(mask)

  # print(counts)
  # print((features[indices, :])[0, :, 0])
  # print((mask[:, :, None])[0, :, 0])

  return points_subsampled, (features[indices, :] * ~torch.isnan(mask[:, :, None])).sum(dim=1) / voxel_point_counts[:, None]

  # return points_subsampled, features[indices[:, 0], :]
  # return points_subsampled, nanmax(features[indices, :] * mask[:, :, None], dim=1)

  # features_subsampled = torch.zeros((len(unique_voxel_ids), features.size(1)), dtype=points.dtype)
  # features_subsampled = torch.full((len(unique_voxel_ids), features.size(1)), -torch.inf, dtype=points.dtype)
  # points_subsampled = torch.zeros((len(unique_voxel_ids), points.size(1)), dtype=points.dtype)

  # for point_index, point_voxel_index in enumerate(point_voxel_indices):
  #   features_subsampled[point_voxel_index, :] += features[point_index, :]
  #   # features_subsampled[point_voxel_id, :] = torch.maximum(features_subsampled[point_voxel_id, :], features[point_index, :])
  #   points_subsampled[point_voxel_index, :] += points[point_index, :]

  # features_subsampled /= voxel_point_counts[:, None]
  # points_subsampled /= voxel_point_counts[:, None]

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



# kernel = create_kernels(
#   dimension=3,
#   kernel_count=1,
#   radius=(2.1 * 0.05),
#   shell_point_counts=[1, 16, 32],
# )[0, :, :]



path = Path('s3dis/Area_1/conferenceRoom_1')
points = np.load(path / 'coord.npy') # dtype: float32, shape: (n, 3)
# colors = np.load(path / 'color.npy') # dtype: uint8, shape: (n, 3)
# labels = np.load(path / 'segment.npy') # dtype: int16, shape: (n, 1)

# write_ply('test_kernel.ply', [kernel.numpy() + points.min(axis=0)], ['x', 'y', 'z'])

print('Subsampling')
points_subsampled = grid_subsample_fast(points, cell_size=0.05)
print('Model')

# points_subsampled = grid_max_pool(torch.tensor(points), torch.ones((points.shape[0], 1)), cell_size=0.05 * 2)[0].detach().numpy()

# # print(points.shape)
# # print(points_subsampled.shape)

# # write_ply('test_default.ply', [points], ['x', 'y', 'z'])
# write_ply('test_subsampled.ply', [points_subsampled], ['x', 'y', 'z'])


torch.autograd.set_detect_anomaly(True)

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

x_half = (points_subsampled[:, 0].max() - points_subsampled[:, 0].min()) * 0.5 + points_subsampled[:, 0].min()

for _ in range(50):
  model.train()
  optimizer.zero_grad()

  model_points, model_features = model(torch.tensor(points_subsampled))
  # break

  loss = ((model_features[model_points[:, 0] > x_half] - 1.0) ** 2).mean() + ((model_features[model_points[:, 0] <= x_half] + 1.0) ** 2).mean()
  print(loss.item())
  # print('features', model.kpconv1.weights.grad)

  loss.backward()
  optimizer.step()


# print(model_features.min(), model_features.max())

# model_features -= model_features.min()
# model_features /= model_features.max()
model_features -= model_features.mean()
model_features /= model_features.std()
model_features = model_features.clamp(-1, 1)

model_features *= 255
model_features = model_features.int()

print(model_features[np.random.randint(0, model_features.shape[0], size=100), 0])

# print(model_points.detach().numpy(), model_features[:, :3].detach().numpy())

write_ply('test_conv.ply', [model_points.detach().numpy(), model_features[:, :3].detach().numpy()], ['x', 'y', 'z', 'red', 'green', 'blue'])
