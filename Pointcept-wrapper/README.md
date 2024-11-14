# KPConvX: Pointcept wrappers

This folder contains wrappers to use KPConvX with the Pointcept library. This allows training KPConvX with multiple GPUs and on multiple datasets.

## Usage

In order to use these wrappers, follow these step-by-step instructions:

- Install Pointcept following the [original intructions](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#installation).

- Copy the files from `Pointcept-wrapper/configs` to the `pointcept/configs` directory.

- Copy the folders from `Pointcept-wrapper/models` in the `pointcept/models` directory.

- Add the following lines to `pointcept/models/__init__.py`
```python
from .kpconvx import *
from .kpnext import *
```

- Compile the cpp wrappers in `models/kpconvx/cpp_wrappers`. 
```bash
cd models/kpconvx/cpp_wrappers/cpp_subsampling
python3 setup.py build_ext --inplace
cd ../cpp_neighbors
python3 setup.py build_ext --inplace
```

At this point you should be able to train a KPConvX model with the Poincept library.

If you clone Poincept in this folder, you can use the following bash commands:

```bash
# Install KPConvX with Pointcept
cp configs/scannet/* Pointcept/configs/scannet/
cp -r models/* Pointcept/pointcept/models/
echo "from .kpconvx import *" >> Pointcept/pointcept/models/__init__.py
echo "from .kpnext import *" >> Pointcept/pointcept/models/__init__.py
cd Pointcept/libs/pointops
python3 setup.py install --user
cd ../../pointcept/models/kpconvx/cpp_wrappers/cpp_subsampling
python3 setup.py build_ext --inplace
cd ../cpp_neighbors
python3 setup.py build_ext --inplace
cd ../../../../..

# Start a training
sh scripts/train.sh -p python -d scannet -c semseg-kpconvx-base -n semseg-kpconvx-base -g 1
#or
sh scripts/train.sh -p python -d scannet -c semseg-kpnext-base -n semseg-kpnext-base -g 1
```




## Details

We provide two different models to use with Pointcept:

### KPNeXt

`kpnext` is our first attempt at using KPConvX convolution with the pointcept library and aimed to be a direct comparison with PointTransformer v2. We used the `point_transformer_v2m2_base` model file and modified it to use kpconvx layers instead of self-attention layers.


### KPConvX

`kpconvx` is a strict conversion of the Standalone model to be used with Poincept training pipeline. Although the backbone operations are identical, the Poincept and the Standalone training pipelines are different, leading to varying final performances. In the paper, we use the Standalone pipeline for S3DIS and the Pointcept pipeline for Scannet

