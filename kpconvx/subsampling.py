import numpy as np


# See https://pastel.hal.science/tel-02458455/document
# Section II.4
def grid_subsampling(
  points: np.ndarray,
  cell_size: np.ndarray | float,
):
  point_voxel_ids = np.floor(points / cell_size)
  unique_voxel_ids = np.unique(point_voxel_ids, axis=0)

  points_subsampled = np.empty((len(unique_voxel_ids), 2), dtype=points.dtype)

  for voxel_index, voxel_id in enumerate(unique_voxel_ids):
    voxel_points = points[(point_voxel_ids == voxel_id).all(axis=-1), :]
    points_subsampled[voxel_index, :] = voxel_points.mean(axis=0)

  return points_subsampled


if __name__ == '__main__':
  from matplotlib import pyplot as plt, ticker

  np.random.seed(0)

  points = np.random.normal(size=(100, 2))
  points_subsgampled = grid_subsampling(points, np.array([0.5, 1.0]))

  print(points.shape, points_subsgampled.shape)

  fig, ax = plt.subplots()
  ax.scatter(points[:, 0], points[:, 1])
  ax.scatter(points_subsgampled[:, 0], points_subsgampled[:, 1])
  ax.grid()
  ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
  plt.show()
