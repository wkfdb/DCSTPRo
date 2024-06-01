# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class VocDataset(CocoDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
               (255, 208, 186), (0, 60, 100), (0, 0, 142), (255, 77, 255),
               (147, 186, 208), (120, 166, 157), (119, 0, 170), (0, 226, 252),
               (182, 182, 255), (0, 0, 230), (220, 20, 60), (3, 95, 161),
               (0, 82, 0), (153, 69, 1), (0, 80, 100), (0, 165, 120)]