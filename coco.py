import os
import cv2
import random
import numpy as np
from copy import deepcopy
from skimage.filters import gaussian
from torchvision.datasets import CocoDetection
from copy_paste import copy_paste_class

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False

@copy_paste_class
class CocoDetectionCP(CocoDetection):
    def __init__(
        self,
        root,
        annFile,
        transforms,
        paste_transforms=None,
        max_paste_objects=None,
        blend=True,
        paste_p=0.5
    ):
        super(CocoDetectionCP, self).__init__(
            root, annFile, None, None, transforms
        )

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids

        self.paste_transforms = paste_transforms
        self.max_paste_objects = max_paste_objects
        self.blend = blend
        self.paste_p = paste_p

    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #convert all of the target segmentations to masks
        masks = []
        bboxes = []
        bbox_classes = []
        for ix, obj in enumerate(target):
            masks.append(self.coco.annToMask(obj))
            bboxes.append(obj['bbox'])
            bbox_classes.append(obj['category_id'])

        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes,
            'bbox_classes': bbox_classes,
        }

        output = self.transforms(**output)

        #remove empty masks
        output['masks'] = list(filter(lambda x: x.sum() > 0, output['masks']))

        assert (len(output['masks']) == len(output['bboxes']))

        return self.transforms(**output)
