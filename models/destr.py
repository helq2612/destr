# ------------------------------------------------------------------------
# DESTR: Object Detection with Split Transformer
# Copyright (c) 2022 Oregon State University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer, gen_sineembed_for_position
from .dcn.deform_conv import DeformConv
from mmcv.cnn import ConvModule

class DESTR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # for two branches
        self.hidden_dim=hidden_dim
        self.feat_channels = self.hidden_dim
        self.stacked_convs = 4
        self.conv_cfg = None
        self.in_channels = hidden_dim
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.conv_bias = 'auto'
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_pos_convs()
        
        
    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_pos_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.pos_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            self.pos_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        
        src, mask = features[-1].decompose()
        assert mask is not None
        bs, num_channels, fh, fw = src.shape
        
        src_proj = self.input_proj(src)

        # I empirically find more supervisions on encoder layers will not help 
        # (because FCOS style supervision will make the feature map pixel isolated while using matching loss)
        num_encoders = 1    
        if num_encoders != 1:
            memory= self.transformer.forward_encoder_all(src_proj, mask, pos[-1])           # [num_encoders*bs, d_model, fh, fw]
        else:
            memory= self.transformer.forward_encoder(src_proj, mask, pos[-1])               # [1*bs, d_model, fh, fw]
        
        fine_pos = pos[-1].flatten(2).permute(2, 0, 1)
        fine_pos = fine_pos * self.transformer.encoder.context_pos_condition(memory.flatten(2).permute(2, 0, 1).contiguous())
        fine_pos = fine_pos.reshape(fh, fw, bs, -1).permute(2, 3, 0, 1).contiguous()
        fine_mask = mask

        # pass to three branches
        d_model = memory.shape[1]
        memory = memory.reshape(1, bs, d_model, fh, fw)[-1*num_encoders:].reshape(-1, d_model, fh, fw)
        cls_memory = memory
        reg_memory = memory
        fine_pos_query = fine_pos
        for cls_layer in self.cls_convs:
            cls_memory = cls_layer(cls_memory)
        for reg_layer in self.reg_convs:
            reg_memory = reg_layer(reg_memory)
        for pos_layer in self.pos_convs:
            fine_pos_query = pos_layer(fine_pos_query)
        fine_pos_query = fine_pos_query.reshape(num_encoders * bs, -1, fh*fw).permute(0, 2, 1).contiguous() 

        # get the coarse prediction, similar to FCOS, but only use one layer
        reg_memory = reg_memory.reshape(num_encoders * bs, -1, fh*fw).permute(0, 2, 1).contiguous()    # [num_encoders * bs, fh*fw, d_model]
        cls_memory = cls_memory.reshape(num_encoders * bs, -1, fh*fw).permute(0, 2, 1).contiguous()    # [num_encoders * bs, fh*fw, d_model]        

        # remove the affect of image zero-padding
        reg_memory     = self.remove_zero_padding(reg_memory,      mask, fh, fw, num_encoders)
        cls_memory     = self.remove_zero_padding(cls_memory,      mask, fh, fw, num_encoders)
        fine_pos_query = self.remove_zero_padding(fine_pos_query,  mask, fh, fw, num_encoders)

        fine_pos_query = fine_pos_query.permute(1, 0, 2).contiguous()               # # fh*fw, num_encoders*bs, d_model

        # get predictions from the last encoder layer output
        reference_points_before_sigmoid = self.transformer.decoder.ref_point_head(fine_pos_query)
        det_reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        det_reference_before_sigmoid = inverse_sigmoid(det_reference_points)
        det_reference_before_sigmoids = torch.cat([det_reference_before_sigmoid for _ in range(num_encoders)], dim=0)
        tmp = self.bbox_embed(reg_memory)  
        tmp[..., :2] += det_reference_before_sigmoids 
        det_outputs_coord = tmp.sigmoid()                                               # [num_encoders*bs, fh*fw, 4] 
        det_outputs_class = self.class_embed(cls_memory)                                # [num_encoders*bs, fh*fw, 91]  
        
        det_outputs_coord1 = det_outputs_coord.reshape(num_encoders, bs, fh*fw, 4)
        det_outputs_class1 = det_outputs_class.reshape(num_encoders, bs, fh*fw, -1)
        
        out1 = dict()
        out1['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for (a, b) in zip(det_outputs_class1[-1*num_encoders:], det_outputs_coord1[-1*num_encoders:])]

        query_feats = torch.cat((cls_memory.reshape(num_encoders, bs, fh*fw, d_model)[-1], 
                                 reg_memory.reshape(num_encoders, bs, fh*fw, d_model)[-1]), dim=-1)                    # [selected_query_embedbs, 2*d_model, fh, fw]
        query_feats = query_feats.permute(1, 0, 2).contiguous()         # fh*fw, bs, 2*d_model

        det_outputs_coord = self.remove_zero_padding(det_outputs_coord, mask, fh, fw, num_encoders)
        det_outputs_class = self.remove_zero_padding(det_outputs_class.sigmoid(), mask, fh, fw, num_encoders)
        det_outputs_coord = det_outputs_coord.reshape(num_encoders, bs, fh*fw, 4)
        det_outputs_class = det_outputs_class.reshape(num_encoders, bs, fh*fw, -1)

        if self.training:
            num_selected_memory = min(self.num_queries, det_outputs_class[-1].shape[1])
        else:
            num_selected_memory = min(self.num_queries, det_outputs_class[-1].shape[1])
        
        valid_num_per_img = mask.flatten(1).eq(0).sum(dim=-1, keepdim=False)[0]
        valid_num_per_img = valid_num_per_img.min()
        num_selected_memory = min(num_selected_memory, valid_num_per_img)
        
        memory_idx = self.topk_batch_select(det_outputs_class[-1], k=num_selected_memory, padding_mask=mask.flatten(1), isTrain=self.training)
        selected_memory = query_feats[memory_idx].reshape(bs, num_selected_memory, -1).permute(1, 0, 2).contiguous()
        selected_memory = selected_memory.detach()

        center_points = det_outputs_coord[-1][..., :2].transpose(0, 1)[memory_idx].reshape(bs, num_selected_memory, -1).permute(1, 0, 2).contiguous()
        center_points = center_points.detach()
        query_pos = gen_sineembed_for_position(center_points, self.hidden_dim)
        selected_ref_points = center_points.transpose(0, 1)

        # get context, only select the last layer output
        fine_memory = memory.reshape(num_encoders, bs, d_model, fh, fw)[-1].flatten(-2).permute(2, 0, 1).contiguous()       # fh*fw, bs, d_model
        fine_mask   = fine_mask.reshape(bs, -1)
        fine_pos = fine_pos.flatten(-2).permute(2, 0, 1).contiguous()                                                       # [fh*fw, bs, d_model]

        hs, reference = self.transformer.forward_decoders_with_queries(tgt=selected_memory, 
                                                                                 memory=fine_memory, 
                                                                                 mask=fine_mask, 
                                                                                 query_embed=query_pos, 
                                                                                 pos_embed=fine_pos,
                                                                                 reference_points=selected_ref_points,
                                                                                 bbox_embed_func=self.bbox_embed)
                                                                                 
        cls_hs, reg_hs = torch.split(hs, [d_model, d_model], dim=-1)                                                        # [num_decoders, bs, fh*fw, d_model] 
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(reg_hs[lvl])                 
            tmp[..., :2] += reference_before_sigmoid       
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = self.class_embed(cls_hs)
        
        out = dict()
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes']  = outputs_coord[-1]
        
        if self.aux_loss:
            out1['aux_outputs'] = out1['aux_outputs'] + self._set_aux_loss(outputs_class, outputs_coord)
        out['aux_outputs']  = out1['aux_outputs']
        return out

    def remove_zero_padding(self, input_feats, mask, fh, fw, lvl=6, padding_val=float(0)):
        # mask: 
        lvl_bs, hw, c = input_feats.shape
        key_padding_mask = mask.flatten(1).repeat(lvl, 1)
        input_feats = input_feats.masked_fill(key_padding_mask.unsqueeze(-1), padding_val)
        return input_feats

    def topk_batch_select(self, x, k=100, padding_mask=None, isTrain=False):
        """
        x, 
        torch.Size([2, 648, 91])
        """
        batch, num, num_classes = x.shape
        default_device = x.device
        if isTrain:
            num_cls = 80
        else:
            num_cls = num_classes
        cls_scores = x.sigmoid()
        max_cls_scores, max_cls_ids= cls_scores[:, :, :num_cls].max(dim=2, keepdim=False)       # [2, 648]

        _, top_k_inds = torch.topk(max_cls_scores, k=k, dim=1, largest=True)
        if padding_mask is not None:
            padding_mask = (1-padding_mask.float()).sum(dim=-1)
            new_topK = []
            for topK_idx, masked in zip(top_k_inds, padding_mask):
                if masked > num:
                    new_topK.append(topK_idx)
                    continue
                temp = torch.flip(topK_idx[:masked.long()], dims=(0,)).repeat(k // masked.long()+1)
                temp = temp[:k]
                temp = torch.cat((topK_idx[:masked.long()], temp[masked.long():]), dim=0)
                new_topK.append(temp)
            top_k_inds = torch.stack(new_topK, dim=0)
        bidex = [(torch.ones(k, device=default_device)*batch_id).long() for batch_id in range(batch)]
        bidex = torch.cat(bidex)
        inds = top_k_inds.reshape(-1)
        selected_idx = (inds, bidex)
        return  selected_idx
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
   
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if i == 0:
                        mini_det_weight = 0.1   #TODO: hard code here, need to move to args
                    else:
                        mini_det_weight = 1.0
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v*mini_det_weight for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _expand(tensor, length):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DESTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers + 2 - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    
    if args.masks:
        losses += ["masks"]

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
