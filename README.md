# Copy-Paste
Unofficial implementation of the copy-paste augmentation from [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/abs/2012.07177v1).

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
    def __init__(self, *args):
        super(SomeVisionDataset, self).__init__(*args)

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
      A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
      A.PadIfNeeded(256, 256, border_mode=0), #constant 0 border
      A.RandomCrop(256, 256),
      A.HorizontalFlip(p=0.5),
      CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1)
    ], bbox_params=A.BboxParams(format="coco")
)
```
