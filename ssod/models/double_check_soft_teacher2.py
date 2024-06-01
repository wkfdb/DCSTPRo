import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply, multiclass_nms
from mmdet.models import DETECTORS, build_detector
import torch.nn.functional as F

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid


from mmdet.models.losses import accuracy

@DETECTORS.register_module()
class DCSTPro(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(DCSTPro, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            # self.unsup_weight=4
            self.unsup_weight = self.train_cfg.unsup_weight

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}

            loss.update(**sup_loss)
            
            
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    # 将无标签数据输入到teacher和studen模型。没有标签怎么计算的loss
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}

            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )
        pseudo_bboxes = self._transform_bbox(
            teacher_info["new_gt_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["new_gt_labels"]
        loss = {}
        
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )

        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal: # False
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        
        unsup_rcnn_cls_loss = self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals, 
                pseudo_bboxes, 
                pseudo_labels, 
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        if unsup_rcnn_cls_loss is not None:
            loss.update(unsup_rcnn_cls_loss)
        
        loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["downsample_feat"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
            )
        )
        
        return loss
    
    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=None,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            # log_image_with_boxes(
            #     "rpn",
            #     student_info["img"][0],
            #     pseudo_bboxes[0][:, :4],
            #     bbox_tag="rpn_pseudo_label",
            #     scores=pseudo_bboxes[0][:, 4],
            #     interval=500,
            #     img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            # )
            return losses, proposal_list
        else:
            return {}, None





    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):

        # candidate_proposals = [res[:, :4] for res in proposal_list]
        candidate_proposals = [torch.cat((res[:, :4],res2[:,:4]),dim=0) for (res,res2) in zip(proposal_list,pseudo_bboxes)]

        rois = bbox2roi(candidate_proposals)
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            candidate_proposals,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            # _scores = torch.cat(_scores, dim=0)
            # _scores = self.teacher.roi_head.double_check_bboxes(
            #     teacher_feat,
            #     teacher_img_metas,
            #     aligned_proposals,
            # )
            selected_bboxes, cls_pseudo_labels, label_weights = self.select_useful_bboxes(candidate_proposals, _scores)
        cls_pseudo_labels = torch.cat(cls_pseudo_labels, dim=0)
        label_weights = torch.cat(label_weights, dim=0)

        if cls_pseudo_labels.size()[0] == 0:
            return None

        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        
        # 计算loss
        loss = self.soft_loss2(
            bbox_results["cls_score"],
            None,
            rois,
            cls_pseudo_labels,
            label_weights,
            reduction_override="mean",
        )
        
        # downsample image loss
        st_bboxes = [bbox / 2.0 for bbox in selected_bboxes]
        st_rois = bbox2roi(st_bboxes)
        downsample_feat = student_info["downsample_feat"]
        st_bbox_results = self.student.roi_head._bbox_forward(downsample_feat, st_rois)
        loss2 = self.soft_loss2(
            st_bbox_results["cls_score"], 
            None,
            st_rois,
            cls_pseudo_labels,
            label_weights,
            reduction_override="mean",
        )
        loss['loss_cls'] = (loss2['loss_cls'] + loss['loss_cls']) * 0.5
        
        
        return loss

    
        
    
    def unsup_rcnn_reg_loss(
        self,
        feat,
        downsample_feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=0.9,
            # score=None,
            # thr=None,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        loss1 = self.soft_loss(
            None, 
            bbox_results["bbox_pred"], 
            rois,
            None, 
            *bbox_targets, 
            reduction_override="mean",
        )
        loss_bbox1 = loss1["loss_bbox"]
        
        
        # downsample image loss
        st_bboxes = [res.bboxes / 2.0 for res in sampling_results]
        st_rois = bbox2roi(st_bboxes)
        # downsample_feat = student_info["downsample_feat"]
        st_bbox_results = self.student.roi_head._bbox_forward(downsample_feat, st_rois)
        # st_bbox_targets = (bbox_targets[0], bbox_targets[1], None, None)
        loss2 = self.soft_loss(
            None, 
            st_bbox_results["bbox_pred"], 
            st_rois,
            None,
            *bbox_targets,
            reduction_override="mean",
        )
        loss_bbox2 = loss2["loss_bbox"]
        
        loss_bbox = (loss_bbox1 + loss_bbox2) * 0.5
        
        
        
        # if len(gt_bboxes[0]) > 0:
        #     log_image_with_boxes(
        #         "rcnn_reg",
        #         student_info["img"][0],
        #         gt_bboxes[0],
        #         bbox_tag="pseudo_label",
        #         labels=gt_labels[0],
        #         class_names=self.CLASSES,
        #         interval=500,
        #         img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #     )

        return {"loss_bbox": loss_bbox}

    
    def select_useful_bboxes(self, proposals, teacher_preds, selected_num = 512, pos_ratio = 0.25):
        selected_bboxes, selected_labels, selected_label_weights = [], [], []

        for proposals_per_img, teacher_preds_per_img in zip(proposals, teacher_preds):
            fg_score = teacher_preds_per_img[:,:-1]
            confidence = torch.topk(fg_score, 3)[0].sum(dim=-1)
            bg_score = teacher_preds_per_img[:,-1]
            
            # confidence = torch.max(fg_score,dim=-1)[0]
            fg_mask = confidence>=0.6

            
            fg_proposals = proposals_per_img[fg_mask]
            if fg_proposals.size(0)>0:
                iou = bbox_overlaps(proposals_per_img, fg_proposals)
                fg_mask = iou.max(-1)[0]>=0.5

            num_expected_neg = selected_num - int(fg_mask.sum().item())
            bg_mask = ~fg_mask
            if bg_mask.sum()>num_expected_neg:
                delete_num = int(bg_mask.sum().item() - num_expected_neg)
                neg_idx = torch.where(bg_mask)[0]
                temp = torch.randperm(neg_idx.numel(), device=neg_idx.device)
                temp = temp[:delete_num]
                del_idx = neg_idx[temp]
               # del_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:delete_num]]
                bg_mask[del_idx] = False

            mask = fg_mask | bg_mask

            selected_bboxes.append(proposals_per_img[mask])
            labels = teacher_preds_per_img[mask]
            label_weights = labels.new_ones(labels.size(0))
            selected_label_weights.append(label_weights)
            
            
            instance_num = labels.size(0)
            dis_dim = labels.size(1)
            logits, idxs = torch.topk(labels, dis_dim)
            top1_idx = idxs[:,0]
            keep_mask = labels>1 
            temp = torch.arange(instance_num).to(labels.device)
            sum_score = labels.new_zeros(instance_num)
            done_mask = sum_score>1
            for i in range(dis_dim):
                sum_score += logits[:,i] 
                maj_idxs = idxs[:,i]
                maj_idxs[done_mask] = top1_idx[done_mask] 
                keep_mask[temp,maj_idxs] = True 
                done_mask = sum_score>=0.8 
                if done_mask.sum()==instance_num:
                    break 
            labels[~keep_mask] = 0
            norm = labels.sum(-1).unsqueeze(1)
            norm[norm==0] = 1
            labels /= norm


            selected_labels.append(labels)
        return selected_bboxes, selected_labels, selected_label_weights
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def soft_loss(
        self,
        cls_score,
        bbox_pred,
        rois,
        _scores,
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        reduction_override=None
    ):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.student.roi_head.bbox_head.loss_cls(
                    cls_score,
                    _scores, 
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.student.roi_head.bbox_head.custom_activation:
                    acc_ = self.student.roi_head.bbox_head.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.student.roi_head.bbox_head.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.student.roi_head.bbox_head.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.student.roi_head.bbox_head.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.student.roi_head.bbox_head.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.student.roi_head.bbox_head.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
    
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def soft_loss2(
        self,
        cls_score,
        bbox_pred,
        rois,
        _scores,
        label_weights,
        reduction_override=None
    ):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.student.roi_head.bbox_head.loss_cls(
                    cls_score,
                    _scores,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
        return losses
    
    def get_sampling_result2(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i][:,:4], gt_bboxes_ignore[i], gt_labels[i]
            )
            
            sampling_result = self.student.roi_head.bbox_sampler.sample2(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )

            sampling_results.append(sampling_result)
        return sampling_results


    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        # return [bt @ at.inverse() for bt, at in zip(b, a)]
        return [bt @ at.cpu().inverse().to(bt.device) for bt, at in zip(b, a)]


    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        # scale distillation
        downsample_img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        # downsample_img = F.interpolate(img, scale_factor=0.5, mode='bilinear')
        # student_info["downsample_img"] = downsample_img
        downsample_feat = self.student.extract_feat(downsample_img)
        student_info["downsample_feat"] = downsample_feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            
            rpn_out = list(self.teacher.rpn_head(feat))
            
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list
        
        
        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        
        gt_bboxes, gt_labels, new_gt_bboxes, new_gt_labels = self.compute_gt_regression_weights(
            feat, img_metas, det_bboxes, det_labels)
        teacher_info["gt_bboxes"] = gt_bboxes
        teacher_info["gt_labels"] = gt_labels
        teacher_info["new_gt_bboxes"] = new_gt_bboxes
        teacher_info["new_gt_labels"] = new_gt_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info


    
    
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            
            box_scale = box[:, 2:4] - box[:, :2]
            
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  
            
            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return new_box[:, :, :4].clone() + offset

        return [_aug_single(box) for box in boxes]
    def compute_gt_regression_weights(self, feat, img_metas, det_bboxes, det_labels):
        """
        return:
        new_gt_boxes, List[Tensor]
        new_gt_labels, List[Tensor]
        regression_weights, List[Tensor]

        """
        
        gt_bboxes_with_score, gt_labels, _ = multi_apply(
            filter_invalid,
            # [bbox[:, :4] for bbox in det_bboxes],
            det_bboxes,
            det_labels,
            [bbox[:, 4] for bbox in det_bboxes],
            thr= self.train_cfg.reg_pseudo_threshold 
        )
        gt_bboxes = [bbox[:, :4] for bbox in gt_bboxes_with_score]
        scores = [bbox[:, 4] for bbox in gt_bboxes_with_score]
        times = 10
        frac = 0.05
        teacher_gt_bboxes_aug = self.aug_box(gt_bboxes, times, frac)
        auged_proposal_list = [auged.reshape(-1, auged.shape[-1]) for auged in teacher_gt_bboxes_aug]
        auged_proposal_list = [torch.cat([gt_bboxes[i], auged_proposal_list[i]], dim=0) for i in range(len(gt_bboxes))]

        boxes, _scores = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        new_gt_bboxes = []
        
        new_gt_labels = []
        for i in range(len(img_metas)):
            if boxes[i].shape[0] > 0:
                box_img_i = boxes[i].reshape(boxes[i].shape[0], -1, 4)
                socre_img_i = _scores[i]
                labels_img_i = gt_labels[i]
                new_gt_bbox = gt_bboxes[i].clone()
                
                for j in range(gt_bboxes[i].shape[0]):
                    inds = torch.arange(j, auged_proposal_list[i].shape[0], gt_bboxes[i].shape[0])
                    regress_box = box_img_i[inds, labels_img_i[inds[0]], :]
                    reg_box_score = socre_img_i[inds, labels_img_i[inds[0]]]
                    weight_mean = True
                    if weight_mean:
                        assert len(reg_box_score.shape) == 1
                        ws = reg_box_score.unsqueeze(dim=1).repeat(1, 4)
                        pos_box_mean = (ws * regress_box).sum(dim=0) / reg_box_score.sum()
                    else:
                        pos_box_mean = regress_box.mean(dim=0)
                    
                    new_gt_bbox[j] = pos_box_mean
                   
                new_gt_bboxes.append(new_gt_bbox)
                
                new_gt_labels.append(labels_img_i.clone())
            else:
                new_gt_bboxes.append(gt_bboxes[i].clone())
                new_gt_labels.append(gt_labels[i].clone())
                
            new_gt_bboxes = [torch.cat([a,b.unsqueeze(1)],dim=-1) for a,b in zip(new_gt_bboxes, scores)]
        return gt_bboxes, gt_labels, new_gt_bboxes, new_gt_labels