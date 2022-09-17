# ------------------------------------------------------------------------
# DESTR: Object Detection with Split Transformer
# Copyright (c) 2022 Oregon State University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#   this one is takes fixed random pair set.
#   But we will use the attention calculated in self-attention. to avoid 
#   increased complexity.
#   this one is working. 
# ------------------------------------------------------------------------
import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util import box_ops
from torch.nn.modules.linear import _LinearWithBias
import numpy as np

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

def gen_sineembed_for_position(pos_tensor, d_model):
    scale = 2 * math.pi
    fd_model = d_model//2
    dim_t = torch.arange(fd_model, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / fd_model)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model* 2)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, att_maps, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs, att_maps, references
    
    def forward_encoder(self, src, mask, pos_embed):
        """
        Only pass to the encoder
        """
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape                                     # [batch, channel, fh, fw]
        src = src.flatten(2).permute(2, 0, 1)                       # [batch, channel, fh*fw] => [fh*fw, batch, channel]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        return memory.permute(1, 2, 0).view(bs, c, h, w)            # [batch, channel, 36, fh*fw]

    def forward_encoder_all(self, src, mask, pos_embed):
        """
        Only pass to the encoder
        """
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape                                     # [batch, channel, fh, fw]
        src = src.flatten(2).permute(2, 0, 1)                       # [batch, channel, fh*fw] => [fh*fw, batch, channel]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memories = self.encoder.forward_all(src, src_key_padding_mask=mask, pos=pos_embed)
        memories = torch.cat([memory.permute(1, 2, 0).view(bs, c, h, w) for memory in memories], dim=0)
        
        return memories            # [num_encoders*batch, channel, 36, fh*fw]

    def forward_decoders(self, memory, mask, query_embed, pos_embed):
        """
        Only pass to the decoder
        """    
        num_context, batch, embed = memory.shape                    # [num_context, batch, embed]   
        query_embed = query_embed.unsqueeze(1).repeat(1, batch, 1)  # [100, 384] -> [100, batch, 384]
        tgt = torch.zeros_like(query_embed)

        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs, references

    def forward_decoders_with_queries(self, tgt, memory, mask, query_embed, pos_embed, reference_points=None, bbox_embed_func=None):
        """
        Only pass to the decoder
        """    
        num_context, batch, embed = memory.shape                    # [num_context, batch, embed]   
        
        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed, reference_points=reference_points, bbox_embed_func=bbox_embed_func)
        return hs, references

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        d_model = encoder_layer.d_model
        self.context_pos_condition = MLP(d_model, d_model, d_model, 2)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            context_pos_condition = self.context_pos_condition(output)
            # context_pos_condition = 1
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*context_pos_condition)

        if self.norm is not None:
            output = self.norm(output)

        return output
        
    def forward_all(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        outputs = []
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            outputs.append(output)
        if self.norm is not None:
            for o in outputs:
                o = self.norm(o)
        return outputs

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                reference_points: Optional[Tensor] = None,
                bbox_embed_func=None):
        output = tgt
        num_queries, bs, num_2channels = tgt.shape
        num_channels = num_2channels//2
        intermediate = []
        
        if reference_points is None:
            reference_points_before_sigmoid = self.ref_point_head(query_pos)    # [num_queries, batch_size, 2]
            reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        else:
            reference_points_before_sigmoid = inverse_sigmoid(reference_points)

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)      # [num_queries, batch_size, 2]
            pos_transformation = self.query_scale(output[...,num_channels:])
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model)     # sinusoidal(sigmoid(s))  
            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation

            # get bboxes for each query
            tmp = bbox_embed_func(output[...,num_channels:])
            
            tmp[..., :2] += reference_points_before_sigmoid.transpose(0, 1) 
            outputs_coord = tmp.sigmoid()   # [num_queries, bs, 4]
            boxes = box_ops.box_cxcywh_to_xyxy(outputs_coord) #[ num_queries, bs, 4]
            boxes = boxes.permute(1, 0, 2)
            outputs_coord_cxcy = outputs_coord[...,:2].permute(1, 0, 2)
            gious = []
            cxcys = []
            for i in range(len(boxes)):
                giou = box_ops.box_iou(boxes[i], boxes[i])[0]
                gious.append(giou)
                cxcy = outputs_coord_cxcy[i, :, 0] + outputs_coord_cxcy[i, :, 1]
                cxcyd = cxcy[:, None] > cxcy[None, :]
                cxcys.append(cxcyd)
            gious = torch.stack(gious, dim=0).detach()
            cxcys = torch.stack(cxcys, dim=0).detach()

            output, att_map_cls, att_map_reg = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           gious=gious, 
                           cxcys=cxcys)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return [torch.stack(intermediate).transpose(1, 2), reference_points]

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
                
        # Decoder Self-Attention
        factor = 2
        self.sa_qcontent_proj = nn.Linear(d_model * factor, d_model * factor)
        self.sa_qpos_proj = nn.Linear(d_model * 1, d_model * 1)
        self.sa_kcontent_proj = nn.Linear(d_model * factor, d_model * factor)
        self.sa_kpos_proj = nn.Linear(d_model * 1, d_model * 1)
        self.sa_v_proj = nn.Linear(d_model * factor, d_model * factor)
        self.self_attn  = MultiheadAttention(d_model * factor,   nhead, dropout=dropout, vdim=d_model* factor)
        self.self_attn2 = MultiheadAttention(d_model * factor*2, nhead, dropout=dropout, vdim=d_model* factor*2)
        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model * factor, d_model * factor)

        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_cls_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)
        self.cross_reg_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)
        
           
        self.nhead = nhead

        # Implementation of Feedforward model
        self.dropout_cls = nn.Dropout(dropout)
        self.dropout_reg = nn.Dropout(dropout)

        self.dropout_pair = nn.Dropout(dropout)

        self.linear_cls1 = nn.Linear(d_model*1, dim_feedforward)
        self.linear_cls2 = nn.Linear(dim_feedforward, d_model*1)
        self.linear_reg1 = nn.Linear(d_model*1, dim_feedforward)
        self.linear_reg2 = nn.Linear(dim_feedforward, d_model*1)

        self.norm1 = nn.LayerNorm(d_model*factor)
        self.norm12 = nn.LayerNorm(d_model*factor)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)


        self.norm_cls2 = nn.LayerNorm(d_model*1)
        self.dropout_cls2 = nn.Dropout(dropout)
        self.norm_cls3 = nn.LayerNorm(d_model*1)
        self.dropout_cls3 = nn.Dropout(dropout)   

        self.norm_reg2 = nn.LayerNorm(d_model*1)
        self.dropout_reg2 = nn.Dropout(dropout)
        self.norm_reg3 = nn.LayerNorm(d_model*1)
        self.dropout_reg3 = nn.Dropout(dropout)  


        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        self.final_simil_cls = Similarity()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    

    def get_paired_objects(self, q, k, v, random_qidx, cxcys, topk=2):
        """
        This method can be simplied by using torch.gather() or torch.index_select()
        """
        n_q, bs, n_2model = q.shape
        default_device = q.device

        bidex = torch.cat([(torch.ones(n_q, device=default_device)*batch_id).long() for batch_id in range(bs)], dim=0)
        bidex = bidex.flatten(0)
        diag = torch.eye((n_q), device=default_device).unsqueeze(0).repeat(bs, 1, 1)

        random_qidx[diag==1] = n_q + 10     # make sure that the diag is the largest one
        _, top_giou_idx = torch.topk(random_qidx, k=topk, dim=-1, sorted=True, largest=True)        # bs, n_q, topk
        secondIoUIdx = top_giou_idx[..., 1].flatten(0)                                              # bs, n_q

        cxcys_idx = torch.gather(cxcys, dim=-1, index=top_giou_idx)
        
        cxcys_idx_1 = (cxcys_idx[..., 1]).flatten(0).long()
        cxcys_idx_2 = 1 - cxcys_idx_1
        
        twoq = torch.stack([q, q[(secondIoUIdx, bidex)].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()], dim=2)
        twok = torch.stack([k, v[(secondIoUIdx, bidex)].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()], dim=2)
        twov = torch.stack([v, v[(secondIoUIdx, bidex)].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()], dim=2)
        
        bidex = torch.cat([(torch.ones(n_q, device=default_device)*batch_id).long() for batch_id in range(bs)], dim=0)
        fidex = torch.cat([(torch.arange(n_q, device=default_device)).long() for batch_id in range(bs)], dim=0)

        twoq1 = twoq[fidex, bidex,cxcys_idx_1].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()
        twoq2 = twoq[fidex, bidex,cxcys_idx_2].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()

        twok1 = twok[fidex, bidex,cxcys_idx_1].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()
        twok2 = twok[fidex, bidex,cxcys_idx_2].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()

        twov1 = twov[fidex, bidex,cxcys_idx_1].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()
        twov2 = twov[fidex, bidex,cxcys_idx_2].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()
        
        twoq3 = torch.cat([twoq1, twoq2], dim=-1)
        twok3 = torch.cat([twok1, twok2], dim=-1)
        twov3 = torch.cat([twov1, twov2], dim=-1)

        return twoq3, twok3, twov3, cxcys_idx_1,bidex, fidex

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     gious=None,
                     cxcys=None):
        # with attention drop out                
        
        
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)        # 
        q_pos = torch.cat((q_pos, q_pos), dim=-1)

        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        k_pos = torch.cat((k_pos, k_pos), dim=-1)

        v = self.sa_v_proj(tgt)

        n_q, bs, n_2model = q_content.shape
        n_model = n_2model//2
        default_device = q_content.device
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2, _ = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask, return_simil=False, gious=None, returnPair=True)

        twoq3, twok3, twov3, cxcys_idx_1, bidex, fidex = self.get_paired_objects(q, k, v, random_qidx=gious, cxcys=cxcys)

        tgt3, _, = self.self_attn2(twoq3, twok3, value=twov3, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask, return_simil=False, gious=None)
        tgt3 = tgt3.reshape(n_q, bs, 2, n_2model)
        tgt3 = tgt3[fidex, bidex,cxcys_idx_1].reshape(bs, n_q, n_2model).permute(1, 0, 2).contiguous()

        coef = 0.5  # TODO: hard code here, need to move to args
        tgt = coef * self.norm1(tgt + self.dropout1(tgt2) ) + (1-coef) * self.norm12(tgt + self.dropout12(tgt3))
        
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        hw, _, _ = memory.shape

        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_2model = q_content.shape
        n_model = n_2model//2

        k_pos = self.ca_kpos_proj(pos)
        
        # As we use miniDet to get initial values, below codes from cdetr is not used.
        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        # if is_first:
        #     q_pos = self.ca_qpos_proj(query_pos)
        #     q = q_content + q_pos
        #     k = k_content + k_pos
        # else:
        #     q = q_content
        #     k = k_content
        q = q_content
        k = k_content
        cls_q, reg_q = torch.split(q, [n_model, n_model], dim=-1)

        cls_q = cls_q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        reg_q = reg_q.view(num_queries, bs, self.nhead, n_model//self.nhead)

        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)

        cls_q = torch.cat([cls_q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        reg_q = torch.cat([reg_q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)
            
        cls_tgt, att_map = self.cross_cls_attn(query=cls_q,
                                key=k,
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)
        reg_tgt, att_map2 = self.cross_reg_attn(query=reg_q,
                                key=k,
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)
        
        cls_tgt = tgt[..., :n_model] + self.dropout_cls2(cls_tgt)
        cls_tgt = self.norm_cls2(cls_tgt)
        cls_tgt2 = self.linear_cls2(self.dropout_cls(self.activation(self.linear_cls1(cls_tgt))))
        cls_tgt = cls_tgt + self.dropout_cls3(cls_tgt2)
        cls_tgt = self.norm_cls3(cls_tgt)

        reg_tgt = tgt[..., n_model:] + self.dropout_reg2(reg_tgt)
        reg_tgt = self.norm_reg2(reg_tgt)
        reg_tgt2 = self.linear_reg2(self.dropout_reg(self.activation(self.linear_reg1(reg_tgt))))
        reg_tgt = reg_tgt + self.dropout_reg3(reg_tgt2)
        reg_tgt = self.norm_reg3(reg_tgt)
        # ---------- End of attention dropout -----------       

        # ========== End of Cross-Attention =============    
        
        tgt = torch.cat((cls_tgt, reg_tgt), dim=-1)

        # easy to use hook to visualize
        simil_cls, simil_all = None, None
        if self.final_simil_cls is not None:
            simil_all, simil_cls, simil_reg = self.final_simil_cls(tgt)

        return tgt, att_map, (simil_cls, simil_all)



    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                gious = None,
                cxcys = None):
        if self.normalize_before:
            raise NotImplementedError
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed,
                                gious=gious, cxcys=cxcys)
        
class Similarity(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, tgt):
        """
        tgt: num_queries, bs, d_2model
        """
        num_queries, bs, d_2model = tgt.shape
        d_model = d_2model // 2
        cls_q, reg_q = torch.split(tgt, [d_model, d_model], dim=-1)
        all_q = tgt.transpose(0, 1)             # bs, num_queries, d_2model
        cls_q = cls_q.transpose(0, 1)           # bs, num_queries, d_model
        reg_q = reg_q.transpose(0, 1)           # bs, num_queries, d_model
        all_simil = torch.bmm(all_q, all_q.transpose(1, 2))
        cls_simil = torch.bmm(cls_q, cls_q.transpose(1, 2))
        reg_simil = torch.bmm(reg_q, reg_q.transpose(1, 2))
        all_simil = F.softmax(all_simil, dim=-1)
        cls_simil = F.softmax(cls_simil, dim=-1)
        reg_simil = F.softmax(reg_simil, dim=-1)
        return all_simil, cls_simil, reg_simil


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
