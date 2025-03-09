import dexc
dexc.install()

from pathlib import Path
import numpy as np

from .ply import write_ply


path = Path('s3dis/Area_1/conferenceRoom_1')
points = np.load(path / 'coord.npy') # dtype: float32, shape: (n, 3)
colors = np.load(path / 'color.npy') # dtype: uint8, shape: (n, 3)
labels = np.load(path / 'segment.npy') # dtype: int16, shape: (n, 1)

# points = cdata[:,0:3].astype(np.float32)
# features = cdata[:, 3:6].astype(np.float32)
# labels = cdata[:, 6:7].astype(np.int32)

write_ply(
  'test.ply',
  [points, colors / 255, labels[:, 0] / 12],
  ['x', 'y', 'z', 'red', 'green', 'blue', 'label']
)
