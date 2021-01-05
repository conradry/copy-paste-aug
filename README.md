# Copy-Paste
Unofficial implementation of the copy-paste augmentation for segmentation and detection tasks:
https://arxiv.org/abs/2012.07177v1.

Setup for torchvision datasets and albumentations. Still a WIP; bugs/features left to hammer out. The target is to have a stable release by January 8th, 2021 (I recommend checking back then before trusting this code as anything other than a reference).

## Current status
- [x] Copy and paste of object instances from instance-separated masks
- [ ] Copy and paste of object instances from a single segmentation mask
- [ ] Albumentations with tensorflow support

## Basic Usage

The idea is to have a standard torchvision dataset that's decorated to add copy-paste functionality.

The dataset class looks like:

```python
from copy_paste import copy_paste_class
from torch.utils.data import Dataset

@copy_paste_class
class SomeVisionDataset(Dataset):
    def __init__(self, ...):
        .....

    def __len__(self):
        return length

    def load_example(self, idx):
        image_data_dict = load_some_data(idx)
        transformed_data_dict = self.transforms(**image_data_dict)
        return transformed_data_dict

```
The only difference from a regular torchvision dataset is the decorator and the ```load_example``` method
instead of ```__getitem__```.

To compose transforms with copy-paste augmentation (bbox params are optional):

```python
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from copy_paste import CopyPaste

transform = A.Compose([
      A.RandomCrop(256, 256, p=1),
      A.OneOf([
        A.Sequential([A.RandomScale(scale_limit=(-0.9, 0), p=1), A.PadIfNeeded(256, 256, border_mode=0)]),
        A.Sequential([A.RandomScale(scale_limit=(0, 1), p=1), A.RandomCrop(256, 256)])
      ]),
      CopyPaste(p=0.5),
      A.Normalize(),
      ToTensorV2(),
    ], bbox_params=A.BboxParams(format="coco")
)
```
