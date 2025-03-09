import math
from typing import Iterable, Literal

import torch


def create_kernels(
  *,
  dimension: Literal[2, 3],
  kernel_count: int,
  radius: float,
  shell_point_counts: Iterable[int], # Including center
  _draw: bool = False,
):
  shell_point_counts_ = torch.tensor(shell_point_counts)
  shell_count = len(shell_point_counts_)
  shell_radii = 2 * torch.arange(shell_count) * radius / (2 * shell_count - 1)

  total_point_count = int(shell_point_counts_.sum())
  start_radii = shell_radii.repeat_interleave(shell_point_counts_)

  match dimension:
    case 2:
      start_angles = torch.rand((kernel_count, total_point_count)) * torch.pi * 2
      positions = start_radii[..., None] * torch.stack([
        torch.cos(start_angles),
        torch.sin(start_angles),
      ], dim=-1)
    case 3:
      theta = torch.rand(kernel_count, total_point_count) * 2 * torch.pi
      phi = (torch.rand(kernel_count, total_point_count) - 0.5) * torch.pi
      positions = start_radii[..., None] * torch.stack([
        torch.cos(theta) * torch.cos(phi),
        torch.sin(theta) * torch.cos(phi),
        torch.sin(phi),
      ], dim=2)

  if _draw:
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

  for step in range(400):
    if _draw:
      ax.clear()

      for i in range(kernel_count):
        ax.scatter(positions[i, :, 0], positions[i, :, 1])

      ax.set_aspect('equal')
      ax.set_xlim(-radius, radius)
      ax.set_ylim(-radius, radius)
      ax.grid()

      plt.draw()
      plt.pause(.01)
      plt.show(block=False)

    diffs = positions[:, :, None, :] - positions[:, None, :, :]
    distances = torch.sqrt((diffs ** 2).sum(dim=-1, keepdim=True))
    forces = diffs / distances ** 3
    gradients = forces.nansum(dim=-2)

    normals = positions / torch.linalg.norm(positions, dim=-1, keepdim=True)
    # Valid ???
    gradients -= (gradients * normals).sum(dim=-1, keepdims=True) * normals
    gradients /= torch.linalg.norm(gradients, dim=-1, keepdim=True)
    gradients[:, 0, :] = 0

    positions += gradients * 0.01 * math.exp(-step * 0.01)

  return positions


if __name__ == '__main__':
  create_kernels(
    dimension=3,
    kernel_count=1,
    radius=3.5,
    shell_point_counts=[1, 16, 32],
    _draw=True,
  )

# create_kernels(
#   dimension=2,
#   kernel_count=1,
#   radius=0.5,
#   shell_point_counts=[1, 8, 16],
# )
