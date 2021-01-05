import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy
from skimage.filters import gaussian

#WHY DOES BBOX FORMAT LOOK OFF?

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

def filter_occluded_bbox(bbox, paste_bboxes):
    #remove bbox if it's entirely occluded
    #for pbox in paste_bboxes:
    pass

def bboxes_copy_paste(bboxes, paste_bboxes):
    if paste_bboxes is not None:
        bboxes = bboxes + paste_bboxes
    return bboxes

def keypoints_copy_paste(keypoints, paste_keypoints, alpha):
    #if alpha:
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

        #number of objects
        if self.pct_objects_paste:
            n_objects = max(1, int(len(masks) * self.pct_objects_paste))
        elif self.max_paste_objects:
            n_objects = min(len(masks), self.max_paste_objects)
        else:
            raise Exception("pct_objects_paste and max_paste_objects cannot both be None.")

        #no objects condition
        if n_objects == 0:
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
            range(0, len(masks)), size=n_objects, replace=False
        )

        #create alpha by combining all the objects into
        #a single binary mask
        masks = [masks[idx] for idx in objs_to_paste]
        alpha = masks[0] > 0
        for mask in masks[1:]:
            alpha += mask > 0

        #only keep chosen subset
        if bboxes:
            bboxes = [bboxes[idx] for idx in objs_to_paste]

        return {
            "paste_img": image,
            "alpha": alpha,
            "paste_msk": None,
            "paste_msks": masks,
            "paste_bboxes": bboxes,
            "paste_keypoints": keypoints,
            "objs_to_paste": objs_to_paste
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

        return img_data

    setattr(dataset_class, '_split_transforms', _split_transforms)
    setattr(dataset_class, '__getitem__', __getitem__)

    return dataset_class
