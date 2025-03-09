import numpy as np
from tqdm import tqdm


# See https://pastel.hal.science/tel-02458455/document
# Section II.4
def grid_subsample(
  points: np.ndarray,
  cell_size: np.ndarray | float,
):
  point_voxel_ids = np.floor(points / cell_size)
  unique_voxel_ids = np.unique(point_voxel_ids, axis=0)

  points_subsampled = np.empty((len(unique_voxel_ids), points.shape[1]), dtype=points.dtype)

  for voxel_index, voxel_id in enumerate(tqdm(unique_voxel_ids)):
    voxel_points = points[(point_voxel_ids == voxel_id).all(axis=-1), :]
    points_subsampled[voxel_index, :] = voxel_points.mean(axis=0)

  return points_subsampled


def grid_subsample_fast(
  points: np.ndarray,
  cell_size: np.ndarray | float,
):
  point_grid_ids = np.floor(points / cell_size).astype(int)
  # point_grid_ids -= point_grid_ids.min(axis=0)
  # x = point_grid_ids[:, 0] + point_grid_ids[:, 1] * (point_grid_ids[:, 0].max() + 1) # + point_grid_ids[:, 2] * (point_grid_ids[:, 0].max() + 1) * (point_grid_ids[:, 1].max() + 1)

  unique_voxel_ids, point_voxel_ids, voxel_point_counts = np.unique(point_grid_ids, axis=0, return_counts=True, return_inverse=True)
  points_subsampled = np.zeros((len(unique_voxel_ids), points.shape[1]), dtype=points.dtype)

  for point_index, point_voxel_id in enumerate(point_voxel_ids):
    points_subsampled[point_voxel_id, :] += points[point_index, :]

  points_subsampled /= voxel_point_counts[:, None]
  return points_subsampled


if __name__ == '__main__':
  from matplotlib import pyplot as plt, ticker

  np.random.seed(0)

  points = np.random.normal(size=(100, 2))
  points_subsgampled = grid_subsample_fast(points, np.array([0.5, 1.0]))

  print(points.shape, points_subsgampled.shape)

  fig, ax = plt.subplots()
  ax.scatter(points[:, 0], points[:, 1])
  ax.scatter(points_subsgampled[:, 0], points_subsgampled[:, 1])
  ax.grid()
  ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
  plt.show()
