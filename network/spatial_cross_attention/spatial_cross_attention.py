import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ops.modules import MSDeformAttn, MSDeformAttn3D

class SpatialCrossAttention(nn.Module):
    # From https://github.com/zhiqi-li/BEVFormer

    def __init__(self, dim=128, dropout=0.1):
        super(SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn3D(embed_dims=dim, num_heads=4, num_levels=1, num_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, query_pos=None, reference_points_cam=None, spatial_shapes=None, bev_mask=None):
        '''
        query: (B, N, C)
        key: (S, M, B, C)
        reference_points_cam: (S, B, N, D, 2), in 0-1
        bev_mask: (S. B, N, D)
        '''
        inp_residual = query
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        S, M, _, _ = key.shape

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros(
            [B, S, max_len, self.dim])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [B, S, max_len, D, 2])

        for j in range(B):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        key = key.permute(2, 0, 1, 3).reshape(
            B * S, M, C)
        value = value.permute(2, 0, 1, 3).reshape(
            B * S, M, C)

        level_start_index = query.new_zeros([1,]).long()
        queries = self.deformable_attention(query=queries_rebatch.view(B*S, max_len, self.dim),
            key=key, value=value,
            reference_points=reference_points_rebatch.view(B*S, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index).view(B, S, max_len, self.dim)

        for j in range(B):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0 
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual