import torch
import torch.nn as nn
import torch.nn.functional as F


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        # print(coords.shape)  [28030, 4]
        # print(coords[:, 0])
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_features_stride'] = 1
        return batch_dict


class PointPillarScatter_spa(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.spa = self.model_cfg.get('IS_SPA', True)
        self.nx, self.ny, self.nz = grid_size
        if self.spa:
            self.pillar_attention_rope = PillarAttentionWithRoPE(num_pillars=self.ny * self.nx, pillar_dim=64,
                                                                 hidden_dim=64)
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = batch_dict['batch_size']
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            sparsity_mask = torch.zeros((self.ny * self.nx), dtype=torch.bool, device=coords.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            if self.spa:
                sparsity_mask[indices] = True
                spatial_feature = self.pillar_attention_rope(spatial_feature.permute(1, 0), sparsity_mask, self.ny,
                                                             self.nx)
                batch_spatial_features.append(spatial_feature.permute(1, 0))
            else:
                batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny,
                                                             self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_features_stride'] = 1
        return batch_dict


class PillarAttentionWithRoPE(nn.Module):
    def __init__(self, num_pillars, pillar_dim, hidden_dim):
        super(PillarAttentionWithRoPE, self).__init__()
        self.num_pillars = num_pillars
        self.pillar_dim = pillar_dim
        self.hidden_dim = hidden_dim

        # MLP layers for generating Q, K, V
        self.q_mlp = nn.Linear(pillar_dim, hidden_dim)
        self.k_mlp = nn.Linear(pillar_dim, hidden_dim)
        self.v_mlp = nn.Linear(pillar_dim, hidden_dim)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pillar_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(pillar_dim)
        self.norm2 = nn.LayerNorm(pillar_dim)

    def apply_rope(self, x, H, W):
        """Apply 2D Rotary Positional Embedding (RoPE) to BEV features."""
        device = x.device
        half_dim = x.shape[-1] // 2
        theta = torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim
        theta = 10000 ** (-theta)  # RoPE scaling factor

        # Create 2D grid for (H, W)
        h_coords = torch.linspace(-1, 1, H, device=device)
        w_coords = torch.linspace(-1, 1, W, device=device)
        meshgrid = torch.stack(torch.meshgrid(h_coords, w_coords), dim=-1)  # (H, W, 2)

        # Apply sine and cosine functions to the coordinates
        h_sin, h_cos = torch.sin(meshgrid[..., 0][:, :, None] * theta), torch.cos(meshgrid[..., 0][:, :, None] * theta)
        w_sin, w_cos = torch.sin(meshgrid[..., 1][:, :, None] * theta), torch.cos(meshgrid[..., 1][:, :, None] * theta)

        # RoPE rotation embedding
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([x1 * h_cos * w_cos - x2 * h_sin * w_sin,
                               x1 * h_sin * w_sin + x2 * h_cos * w_cos], dim=-1)
        return x_rotated

    def forward(self, pillar_features, sparsity_mask, H, W):
        # Apply RoPE to the BEV features and add to original features
        rope_features = self.apply_rope(pillar_features.view(H, W, -1), H, W)
        pillar_features = pillar_features.view(H, W, -1) + rope_features

        # Flatten BEV features
        pillar_features = pillar_features.view(H * W, -1)

        # Gather non-empty pillars based on sparsity mask
        non_empty_pillars1 = pillar_features[sparsity_mask]
        non_empty_pillars = self.norm1(non_empty_pillars1)
        # Linear projections to get Q, K, V
        Q = self.q_mlp(non_empty_pillars)  # (p, E)
        K = self.k_mlp(non_empty_pillars)  # (p, E)
        V = self.v_mlp(non_empty_pillars)  # (p, E)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (p, p)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute attended features
        attended_features = torch.matmul(attention_weights, V)  # (p, E)
        attended_features1 = attended_features + non_empty_pillars1

        # Pass through FFN
        updated_pillars = self.ffn(self.norm2(attended_features1))  # (p, C)

        updated_pillars = updated_pillars + attended_features1
        # Apply layer normalization

        # Scatter back to original positions
        output_pillars = pillar_features.clone()
        output_pillars[sparsity_mask] = updated_pillars

        return output_pillars  # Reshape back to (H, W, C)
