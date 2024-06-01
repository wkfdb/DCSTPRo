from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
import torch
import torch.nn as nn
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes)


@HEADS.register_module()
class StandardRoIHead2(StandardRoIHead):


    def double_check_bboxes(
    	self,
        x,
        img_metas,
        proposals,
    ):

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            return [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                det_label = rois[i].new_zeros((0, self.bbox_head.fc_cls.out_features))
            else:
                det_label = self.bbox_head.get_bboxes_for_check(
                    proposals[i],
                    cls_score[i],  
                )
            det_labels.append(det_label)
        return det_labels

    