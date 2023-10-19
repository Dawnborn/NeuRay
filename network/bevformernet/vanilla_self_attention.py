import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ops.modules import MSDeformAttn, MSDeformAttn3D

class VanillaSelfAttention(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim 
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=4, n_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, query_pos=None):
        '''
        query: (B, N, C)
        query: (B, N, C)
        '''
        inp_residual = query.clone()

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        device = query.device
        Z, X = 200, 200
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z-0.5, Z, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X-0.5, X, dtype=torch.float, device=query.device)
        )
        ref_z = ref_z.reshape(-1)[None] / Z
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_z, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2) # (B, N, 1, 2)

        B, N, C = query.shape
        input_spatial_shapes = query.new_zeros([1,2]).long()
        input_spatial_shapes[:] = 200
        input_level_start_index = query.new_zeros([1,]).long()
        queries = self.deformable_attention(query, reference_points, query.clone(), 
            input_spatial_shapes, input_level_start_index)

        queries = self.output_proj(queries)

        return self.dropout(queries) + inp_residual