from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from torch import nn
import torch

from .kernel import create_kernels


class KPConv(nn.Module):
  def __init__(
    self,
    kernel: torch.Tensor,
    neighbor_count: int,
    input_feature_count: int,
    output_feature_count: int,
  ):
    super().__init__()

    self.neighbor_count = neighbor_count

    # (K, d)
    self.kernel = kernel

    # (K, C, O)
    v = torch.randn(self.kernel.size(0), input_feature_count, output_feature_count)
    v[...] = -1.0
    v[self.kernel[:, 0] > 0, ...] = 1.0
    # self.weights = nn.Parameter(torch.randn(self.kernel.size(0), input_feature_count, output_feature_count))
    self.weights = nn.Parameter(v)

  # G = number of support points
  # Q = number of query points
  # K = number of kernel points
  def forward(
    self,
    query_points: torch.Tensor, # (Q, d)
    support_points: torch.Tensor, # (G, d)
    support_features: torch.Tensor, # (G, C)
  ):
    print(f'Q={query_points.size(0)}')
    print(f'G={support_points.size(0)}')
    print(f'K={self.kernel.size(0)}')
    print(f'C={support_features.size(1)}')
    print(f'O={self.weights.size(2)}')
    print(f'H={self.neighbor_count}')

    tree = KDTree(support_points)

    # Shape: (len(query_points), neighbor_count)
    # Shape: (Q, H)
    neighbor_indices = tree.query(query_points, k=self.neighbor_count, return_distance=False)

    # Shape: (Q, H, d)
    neighbor_points = query_points[neighbor_indices, :]

    # Shape: (Q, H, C)
    neighbor_features = support_features[neighbor_indices, :]

    # (Q, H, K)
    distances = torch.sqrt(((neighbor_points[:, :, None, :] - query_points[:, None, None, :] - self.kernel) ** 2).sum(dim=-1))

    # (Q, H, K)
    sigma = 1.0
    influences = torch.relu(1 - distances / sigma)

    result = np.einsum('qhk, kco, qhc -> qo', influences, self.weights, neighbor_features)

    return result


if __name__ == '__main__':
  torch.random.manual_seed(0)
  kernel = create_kernels(
    dimension=2,
    kernel_count=1,
    radius=0.3,
    shell_point_counts=[1, 8, 16],
  )[0, :, :]

  # cloud = torch.Tensor([
  #   [0, 0],
  #   [0, 1],
  #   [0, 2],
  #   [0, 3],
  #   [1, 3],
  #   [2, 3],
  #   [3, 3],
  #   [4, 3],
  #   [4, 2],
  #   [4, 1],
  # ])

  def line(start, end):
    return torch.stack([
      torch.linspace(start[0], end[0], 20),
      torch.linspace(start[1], end[1], 20),
    ], dim=-1)

  # cloud = torch.stack([
  #   torch.linspace(-1, 1, 20),
  #   torch.zeros(20),
  # ], dim=-1)

  # cloud = torch.cat([
  #   line((-1, -1), (-1, 1)),
  #   line((-1, 1), (1, 1)),
  #   line((1, 1), (1, -1)),
  # ])

  cloud = torch.cat([
    line((x, -1), (x, 1)) for x in torch.linspace(-1, 1, 20)
  ])

  # cloud = torch.randn(50, 2)

  model = KPConv(
    kernel=kernel,
    input_feature_count=1,
    neighbor_count=10,
    output_feature_count=1,
  )

  with torch.no_grad():
    x = model(cloud, cloud, torch.ones((cloud.size(0), 1)))
    # print(x)

    fig, ax = plt.subplots()
    # ax.scatter(cloud[:, 0], cloud[:, 1], cmap='bwr')
    ax.scatter(cloud[:, 0], cloud[:, 1], c=x[:, 0], cmap='RdYlBu')

    p = cloud[0] + kernel
    ax.scatter(p[:, 0], p[:, 1], alpha=0.5, c=model.weights[:, 0, 0], cmap='bwr', marker='x')

    plt.show()

  # path = Path('s3dis/Area_1/conferenceRoom_1')
  # points = np.load(path / 'coord.npy') # dtype: float32, shape: (n, 3)
