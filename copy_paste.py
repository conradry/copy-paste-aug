import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy
from skimage.filters import gaussian
from torchvision.datasets import CocoDetection

min_keypoints_per_image = 10

#TODO: "transform" in compose that indicates position to split
#TODO: create class wrapper for copy paste
#TODO: create tensorflow aug_fn wrapper
#TODO: code for removing empty mask
#TODO: handling completely occluded bounding boxes
#TODO: code for visualizing the resultant masks and bounding boxes

def copy_paste(img_data, paste_img_data, blend=True):
    if paste_img_data is None:
        return img_data
    else:
        alpha = np.zeros_like(paste_img_data['masks'][0]).astype(np.float32)
        for mask in paste_img_data['masks']:
            mask = mask.astype(np.float32)
            if blend:
                mask = gaussian(mask, sigma=3, preserve_range=True)

            alpha += mask

        #alpha = np.clip(alpha[..., None], 0, 1)
        alpha = alpha[..., None]
        image = img_data['image']
        paste_image = paste_img_data['image']
        image = (paste_image * alpha + image * (1 - alpha)).astype(np.uint8)

    img_data['image'] = image
    img_data['masks'].extend(paste_img_data['masks'])
    img_data['bboxes'].extend(paste_img_data['bboxes'])
    img_data['bbox_classes'].extend(paste_img_data['bbox_classes'])

    return img_data

class CopyPaste:
    def __init__(
        self,
        blend=True,
        blend_sigma=3,
        pct_objects_paste=0.1,
        max_paste_objects=None,
        p=0.5,
        image_key = 'image',
        mask_key = 'masks'
    ):
        self.blend = blend
        self.blend_sigma = blend_sigma
        self.pct_objects_paste = pct_objects_paste
        self.max_paste_objects = max_paste_objects
        self.p = p
        self.image_key = image_key
        self.mask_key = mask_key

    def get_class_fullname(self):
        return 'copypaste.CopyPaste'

    def __call__(self, img_data, paste_img_data):
        #select a subset of objects, up to max
        n_objects = len(paste_img_data['bboxes'])
        if n_objects == 0:
            return img_data
        else:
            n_paste = random.randint(1, n_objects)
            objs_to_paste = np.random.choice(
                range(0, n_objects), size=n_paste, replace=False
            )

            paste_img_data['masks'] = [paste_img_data['masks'][i] for i in objs_to_paste]
            paste_img_data['bboxes'] = [paste_img_data['bboxes'][i] for i in objs_to_paste]
            paste_img_data['bbox_classes'] = [paste_img_data['bbox_classes'][i] for i in objs_to_paste]

        #paste_img_data = None
        return copy_paste(img_data, paste_img_data, True)

def copy_paste_class(dataset_class):
    def _split_transforms(self):
        split_index = None
        for ix, tf in enumerate(list(self.transforms.transforms)):
            if tf.get_class_fullname() == 'copypaste.CopyPaste':
                split_index = ix

        if split_index is not None:
            tfs = list(self.transforms.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index+1:]

            #replicate the other augmentation parameters
            bbox_params = None
            keypoint_params = None
            if 'bboxes' in self.transforms.processors:
                bbox_params = self.transforms.processors['bboxes'].params
            if 'keypoints' in self.transforms.processors:
                keypoint_params = self.transforms.processors['keypoints'].params

            additional_targets = self.transforms.additional_targets

            #recreate transforms
            self.transforms = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets)
            self.copy_paste = copy_paste
        else:
            self.copy_paste = None
            self.post_transforms = None

    def __getitem__(self, idx):
        #split transforms if it hasn't been done already
        if not hasattr(self, 'post_transforms'):
            self._split_transforms()

        img_data = self.load_example(idx)
        if self.post_transforms is not None:
            paste_idx = random.randint(0, self.__len__())
            paste_img_data = self.load_example(paste_idx)

            img_data = self.copy_paste(img_data, paste_img_data)
            img_data = self.post_transforms(**img_data)

        return img_data

    setattr(dataset_class, '_split_transforms', _split_transforms)
    setattr(dataset_class, '__getitem__', __getitem__)

    return dataset_class
