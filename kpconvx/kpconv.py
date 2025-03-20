import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from torch import nn

from .kernel import create_kernels


class KPConv(nn.Module):
  def __init__(
    self,
    *,
    input_feature_count: int,
    kernel: torch.Tensor,
    neighbor_count: int,
    output_feature_count: int,
    sigma: float = 1.0,
    use_one_kernel_point_per_neighbor: bool = True,
  ):
    super().__init__()

    self.sigma = sigma
    self.neighbor_count = neighbor_count

    # (K, d)
    self.kernel = kernel

    if use_one_kernel_point_per_neighbor:
      self.kernel_tree = KDTree(kernel)
    else:
      self.kernel_tree = None

    # (K, C, O)
    # v = torch.randn(self.kernel.size(0), input_feature_count, output_feature_count)
    # v[...] = -1.0
    # v[self.kernel[:, 0] > 0, ...] = 1.0
    # self.weights = nn.Parameter(v)

    self.weights = nn.Parameter(torch.randn(self.kernel.size(0), input_feature_count, output_feature_count))

  # G = number of support points
  # Q = number of query points
  # K = number of kernel points
  def forward(
    self,
    query_points: torch.Tensor, # (Q, d)
    support_points: torch.Tensor, # (G, d)
    support_features: torch.Tensor, # (G, C)
  ):
    d = query_points.size(1)

    # print(f'Q={query_points.size(0)}')
    # print(f'G={support_points.size(0)}')
    # print(f'K={self.kernel.size(0)}')
    # print(f'C={support_features.size(1)}')
    # print(f'O={self.weights.size(2)}')
    # print(f'H={self.neighbor_count}')

    support_tree = KDTree(support_points)

    # Shape: (Q, H)
    neighbor_indices = support_tree.query(query_points, k=self.neighbor_count, return_distance=False)

    # Shape: (Q, H, d)
    neighbor_points = query_points[neighbor_indices, :]

    # Shape: (Q, H, C)
    neighbor_features = support_features[neighbor_indices, :]

    if self.kernel_tree is not None:
      # (Q, H, 1)
      diff = neighbor_points - query_points[:, None, :]
      distances, kernel_point_indices = self.kernel_tree.query(diff.view(-1, d), k=1)

      # (Q, H)
      distances = torch.tensor(distances.reshape(neighbor_indices.shape))
      kernel_point_indices = torch.tensor(kernel_point_indices.reshape(neighbor_indices.shape))

      # (Q, H)
      influences = torch.relu(1 - distances / self.sigma)

      return torch.einsum('qh, qhco, qhc -> qo', influences, self.weights[kernel_point_indices, :, :], neighbor_features)
    else:
      # (Q, H, K)
      distances = torch.sqrt(((neighbor_points[:, :, None, :] - query_points[:, None, None, :] - self.kernel) ** 2).sum(dim=-1))

      # (Q, H, K)
      influences = torch.relu(1 - distances / self.sigma)

      # (Q, O)
      return torch.einsum('qhk, kco, qhc -> qo', influences, self.weights, neighbor_features)


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

  # model = KPConvSingleKernelPoint(
  model = KPConv(
    input_feature_count=1,
    kernel=kernel,
    neighbor_count=10,
    output_feature_count=1,
    sigma=1.0,
    use_one_kernel_point_per_neighbor=True,
  )

  with torch.no_grad():
    x = model(cloud, cloud, torch.ones((cloud.size(0), 1)))
    # print(x)

    fig, ax = plt.subplots()
    # ax.scatter(cloud[:, 0], cloud[:, 1], cmap='bwr')
    ax.scatter(cloud[:, 0], cloud[:, 1], c=x[:, 0], cmap='RdYlBu')

    p = cloud[0] + kernel
    ax.scatter(p[:, 0], p[:, 1], alpha=0.5, c=model.weights[:, 0, 0], cmap='RdYlBu', marker='x')

    plt.show()

  # path = Path('s3dis/Area_1/conferenceRoom_1')
  # points = np.load(path / 'coord.npy') # dtype: float32, shape: (n, 3)
