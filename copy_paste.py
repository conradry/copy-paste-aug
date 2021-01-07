import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy
from skimage.filters import gaussian

#BETTER HANDLING OF OCCLUDED BOUNDING BOXES

def image_copy_paste(img, paste_img, alpha, blend=True, sigma=3):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]
        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)

    return img

def mask_copy_paste(msk, paste_msk, alpha):
    raise NotImplementedError

def masks_copy_paste(msks, paste_msks, alpha):
    if alpha is not None:
        #eliminate pixels that will be pasted over
        msks = [
            np.logical_and(msk, np.logical_xor(msk, alpha)).astype(np.uint8) for msk in msks
        ]
        msks.extend(paste_msks)

    return msks

def adjust_occluded_bbox(bbox, paste_bboxes, bbox_format="albumentations"):
    assert(bbox_format == "albumentations"), \
    "Only albumentations bbox format is supported!"

    x1, y1, x2, y2 = bbox[:4]
    tail = bbox[4:]
    for pbox in paste_bboxes:
        px1, py1, px2, py2 = pbox[:4]

        #top edge occluded
        if all([x1 >= px1, x2 <= px2, y1 >= py1, y1 <= py2]):
            y1 = min(y2, py2)

        #bottom edge occluded
        if all([x1 >= px1, x2 <= px2, y2 >= py1, y2 <= py2]):
            y2 = max(y1, py1)

        #left edge occluded
        if all([y1 >= py1, y2 <= py2, x1 >= px1, x1 <= px2]):
            x1 = min(x2, px2)

        #right edge occluded
        if all([y1 >= py1, y2 <= py2, x2 >= px1, x2 <= px2]):
            x2 = max(x1, px1)

    #entirely occluded boxes collapse to have no area
    #they look like (x2, x2, y2, y2)
    return (x1, y1, x2, y2) + tail

def adjust_occluded_bbox(bbox, paste_bboxes, bbox_format="albumentations"):
    assert(bbox_format == "albumentations"), \
    "Only albumentations bbox format is supported!"

    x1, y1, x2, y2 = bbox[:4]
    tail = bbox[4:]
    for pbox in paste_bboxes:
        px1, py1, px2, py2 = pbox[:4]

        #top edge occluded
        if all([x1 >= px1, x2 <= px2, y1 >= py1, y1 <= py2]):
            y1 = min(y2, py2)

        #bottom edge occluded
        if all([x1 >= px1, x2 <= px2, y2 >= py1, y2 <= py2]):
            y2 = max(y1, py1)

        #left edge occluded
        if all([y1 >= py1, y2 <= py2, x1 >= px1, x1 <= px2]):
            x1 = min(x2, px2)

        #right edge occluded
        if all([y1 >= py1, y2 <= py2, x2 >= px1, x2 <= px2]):
            x2 = max(x1, px1)

    #entirely occluded boxes collapse to have no area
    #they look like (x2, x2, y2, y2)
    return (x1, y1, x2, y2) + tail

def bboxes_copy_paste(bboxes, paste_bboxes):
    if paste_bboxes is not None:
        adjusted_bboxes = [adjust_occluded_bbox(bbox, paste_bboxes) for bbox in bboxes]

        #adjust paste_bboxes mask indices to avoid overlap
        if adjusted_bboxes:
            max_mask_index = 1 + max([b[-1] for b in adjusted_bboxes])
        else:
            max_mask_index = 0

        paste_mask_indices = [max_mask_index + ix for ix in range(len(paste_bboxes))]
        adjusted_paste_bboxes = []
        for mi, pbox in zip(paste_mask_indices, paste_bboxes):
            adjusted_paste_bboxes.append(pbox[:-1] + tuple([mi]))

        bboxes = adjusted_bboxes + adjusted_paste_bboxes

    return bboxes

def keypoints_copy_paste(keypoints, paste_keypoints, alpha):
    #remove occluded keypoints
    if alpha is not None:
        visible_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            tail = kp[2:]
            if alpha[int(y), int(x)] == 0:
                visible_keypoints.append(kp)

        keypoints = visible_keypoints + paste_keypoints

    return keypoints

class CopyPaste(A.DualTransform):
    def __init__(
        self,
        blend=True,
        sigma=3,
        pct_objects_paste=0.1,
        max_paste_objects=None,
        p=0.5,
        always_apply=False
    ):
        super(CopyPaste, self).__init__(always_apply, p)
        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste
        self.max_paste_objects = max_paste_objects
        self.p = p
        self.always_apply = always_apply

    @staticmethod
    def get_class_fullname():
        return 'copypaste.CopyPaste'

    @property
    def targets_as_params(self):
        return [
            "paste_image",
            #"paste_mask",
            "paste_masks",
            "paste_bboxes",
            #"paste_keypoints"
        ]

    def get_params_dependent_on_targets(self, params):
        image = params["paste_image"]
        masks = None
        if "paste_mask" in params:
            #handle a single segmentation mask with
            #multiple targets
            #nothing for now.
            raise NotImplementedError
        elif "paste_masks" in params:
            masks = params["paste_masks"]

        assert(masks is not None), "Masks cannot be None!"

        bboxes = params.get("paste_bboxes", None)
        keypoints = params.get("paste_keypoints", None)

        #number of objects: n_bboxes <= n_masks because of automatic removal
        n_objects = len(bboxes) if bboxes is not None else len(masks)

        #paste all objects if no restrictions
        n_select = n_objects
        if self.pct_objects_paste:
            n_select = int(n_select * self.pct_objects_paste)
        if self.max_paste_objects:
            n_select = min(n_select, self.max_paste_objects)

        #no objects condition
        if n_select == 0:
            return {
                "paste_img": None,
                "alpha": None,
                "paste_msk": None,
                "paste_msks": None,
                "paste_bboxes": None,
                "paste_keypoints": None,
                "objs_to_paste": []
            }

        #select objects
        objs_to_paste = np.random.choice(
            range(0, n_objects), size=n_select, replace=False
        )

        #take the bboxes
        if bboxes:
            bboxes = [bboxes[idx] for idx in objs_to_paste]

            #the last label in bboxes is the index of corresponding mask
            mask_indices = [bbox[-1] for bbox in bboxes]

        #create alpha by combining all the objects into
        #a single binary mask
        masks = [masks[idx] for idx in mask_indices]
        alpha = masks[0] > 0
        for mask in masks[1:]:
            alpha += mask > 0

        return {
            "paste_img": image,
            "alpha": alpha,
            "paste_msk": None,
            "paste_msks": masks,
            "paste_bboxes": bboxes,
            "paste_keypoints": keypoints
        }

    @property
    def ignore_kwargs(self):
        return [
            "paste_image",
            "paste_mask",
            "paste_masks",
        ]

    def apply_with_params(self, params, force_apply=False, **kwargs):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None and key not in self.ignore_kwargs:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def apply(self, img, paste_img, alpha, **params):
        return image_copy_paste(
            img, paste_img, alpha, blend=self.blend, sigma=self.sigma
        )

    def apply_to_mask(self, msk, paste_msk, alpha, **params):
        return mask_copy_paste(msk, paste_msk, alpha)

    def apply_to_masks(self, msks, paste_msks, alpha, **params):
        return masks_copy_paste(msks, paste_msks, alpha)

    def apply_to_bboxes(self, bboxes, paste_bboxes, **params):
        return bboxes_copy_paste(bboxes, paste_bboxes)

    def apply_to_keypoints(self, keypoints, paste_keypoints, alpha, **params):
        raise NotImplementedError
        #return keypoints_copy_paste(keypoints, paste_keypoints, alpha)

    def get_transform_init_args_names(self):
        return (
            "blend",
            "sigma",
            "pct_objects_paste",
            "max_paste_objects"
        )

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
            paste_additional_targets = {}
            if 'bboxes' in self.transforms.processors:
                bbox_params = self.transforms.processors['bboxes'].params
                paste_additional_targets['paste_bboxes'] = 'bboxes'
                if self.transforms.processors['bboxes'].params.label_fields:
                    msg = "Copy-paste does not support bbox label_fields! "
                    msg += "Expected bbox format is (a, b, c, d, label_field)"
                    raise Exception(msg)
            if 'keypoints' in self.transforms.processors:
                keypoint_params = self.transforms.processors['keypoints'].params
                paste_additional_targets['paste_keypoints'] = 'keypoints'
                if keypoint_params.label_fields:
                    raise Exception('Copy-paste does not support keypoint label fields!')

            if self.transforms.additional_targets:
                raise Exception('Copy-paste does not support additional_targets!')

            #recreate transforms
            self.transforms = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets=None)
            self.copy_paste = A.Compose(
                [copy_paste], bbox_params, keypoint_params, additional_targets=paste_additional_targets
            )
        else:
            self.copy_paste = None
            self.post_transforms = None

    def __getitem__(self, idx):
        #split transforms if it hasn't been done already
        if not hasattr(self, 'post_transforms'):
            self._split_transforms()

        img_data = self.load_example(idx)
        if self.copy_paste is not None:
            paste_idx = random.randint(0, self.__len__())
            paste_img_data = self.load_example(paste_idx)
            for k in list(paste_img_data.keys()):
                paste_img_data['paste_' + k] = paste_img_data[k]
                del paste_img_data[k]

            img_data = self.copy_paste(**img_data, **paste_img_data)
            img_data = self.post_transforms(**img_data)
            img['paste_index'] = paste_idx

        return img_data

    setattr(dataset_class, '_split_transforms', _split_transforms)
    setattr(dataset_class, '__getitem__', __getitem__)

    return dataset_class
