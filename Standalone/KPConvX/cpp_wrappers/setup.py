import os

import numpy as np
from setuptools import setup
from setuptools.extension import Extension


setup(
  ext_modules=[
    Extension(
      'cpp_wrappers.cpp_neighbors',
      [
        'cpp_utils/cloud/cloud.cpp',
        'cpp_neighbors/neighbors/neighbors.cpp',
        'cpp_neighbors/wrapper.cpp',
      ],
      include_dirs=[np.get_include()],
      extra_compile_args=['-std=c++11', '-D_GLIBCXX_USE_CXX11_ABI=0'],
    ),
    Extension(
      'cpp_wrappers.cpp_subsampling',
      [
        'cpp_utils/cloud/cloud.cpp',
        'cpp_subsampling/grid_subsampling/grid_subsampling.cpp',
        'cpp_subsampling/fps_subsampling/fps_subsampling.cpp',
        'cpp_subsampling/wrapper.cpp',
      ],
      include_dirs=[np.get_include()],
      extra_compile_args=['-std=c++11', '-D_GLIBCXX_USE_CXX11_ABI=0'],
    ),
  ],
  packages=[],
)
