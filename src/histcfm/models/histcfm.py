"""Formal cell-level HistCFM model.

Derived from the cell-level framework of SydneyBioX/GHIST:
https://github.com/SydneyBioX/GHIST

Modified for HistCFM on 2026-08-12 to provide conditional flow-based
expression prediction, precomputed UNI-guided auxiliary regularization,
SONRM-facing state, and cell-level iterative inference. Distributed under
GNU GPL version 3 only; see the repository ``LICENSE`` file.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features.uni import UniFeatureProvider
from ..graph import build_knn_graph
from .backbone import Backbone
from .components import CrossAttention, Embed, MLP, MLPSoftmax
from .flow import FlowExpressionModel
from .fusion import FusionFiLM, FusionGate


def safe_shape(tensor):
    return tensor.shape if tensor is not None else 'None'




class HistCFM(nn.Module):
    def __init__(
        self,
        n_classes,
        n_genes,
        emb_dim,
        device,
        n_ref,
        use_avgexp,
        use_celltype,
        use_neighb,
        in_channels=3,
        use_flow_expr=True,
        flow_hidden_dim=512,
        flow_layers=4,
        flow_k_neighbors=16,
        flow_use_zinb=False,
        flow_steps=5,
        flow_noise_train_sigma=0.02,
        flow_noise_infer_sigma=0.05,
        uni_enable=False,
        uni_mode="precomputed",
        uni_index_path=None,
        uni_features_path=None,
        uni_dim=1024,
        fusion_type="gate",
        fusion_hidden=256,
        fusion_dropout=0.0,
    ):
        super(HistCFM, self).__init__()
        n_classes_backbone = n_classes + 1 if use_celltype else 2
        self.cnn = Backbone(
            n_channels=in_channels,
            bilinear=True,
            is_deconv=True,
            is_batchnorm=True,
            n_classes=n_classes_backbone,
        )
        dim_fv = 384 * 2
        num_heads = 8
        self.hidden_size = emb_dim
        self.device = device
        self.use_avgexp = use_avgexp
        self.use_celltype = use_celltype
        self.use_neighb = use_neighb
        self.n_classes = n_classes
        self.n_genes = n_genes
        self.n_ref = n_ref if n_ref is not None else 0
        self.use_flow_expr = use_flow_expr
        self.flow_steps = flow_steps
        self.flow_k_neighbors = flow_k_neighbors
        self.flow_noise_train_sigma = flow_noise_train_sigma
        self.flow_noise_infer_sigma = flow_noise_infer_sigma
        self.embed_hist = Embed(dim_fv, self.hidden_size)
        if self.use_neighb:
            self.estimate_comp = MLPSoftmax(self.hidden_size, self.hidden_size, n_classes)
            self.refine_expr = CrossAttention(n_classes, n_genes, self.hidden_size, num_heads=num_heads)
            self.refine_expr_immune = CrossAttention(n_classes, n_genes, self.hidden_size, num_heads=num_heads)
            self.refine_expr_invasive = CrossAttention(n_classes, n_genes, self.hidden_size, num_heads=num_heads)
        if self.use_celltype:
            self.mlp_hist = MLP(self.hidden_size, self.hidden_size, n_classes)
            self.mlp_genes = MLP(n_genes, self.hidden_size, n_classes)
        self.flow_expr = FlowExpressionModel(n_genes=n_genes, feat_dim=emb_dim, hidden_dim=flow_hidden_dim, n_layers=flow_layers, k_neighbors=flow_k_neighbors, use_zinb=flow_use_zinb)
        self.uni_provider = UniFeatureProvider(enable=uni_enable, mode=uni_mode, index_path=uni_index_path, features_path=uni_features_path, uni_dim=uni_dim, device=self.device)
        if fusion_type == "film":
            self.fusion = FusionFiLM(self.hidden_size, uni_dim, self.hidden_size, hidden_dim=fusion_hidden, dropout=fusion_dropout)
            self.fusion_type = "film"
        else:
            self.fusion = FusionGate(self.hidden_size, uni_dim, self.hidden_size, hidden_dim=fusion_hidden, dropout=fusion_dropout)
            self.fusion_type = "gate"
        self.uni_blend_alpha = 1.0
        self.last_embeddings = None
        self.last_coords = None
        self.uni_hint_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, n_genes))

    def forward(
        self,
        x_hist,
        nuclei_mask,
        n_cells,
        ref_orig,
        batch_ct=None,
        batch_expr=None,
        patch_ids=None,
        patch_keys=None,
        do_st_mlp=True,
        stage_pretrain=False,
    ):
        out_map, hd1, h1 = self.cnn(x_hist)
        n_cells_total = torch.sum(n_cells)
        all_fv = torch.zeros((n_cells_total, 2 * (hd1.shape[1] + h1.shape[1]))).to(self.device)
        all_area = torch.zeros(n_cells_total).to(self.device)
        patch_area_hd1 = hd1.shape[2] * hd1.shape[3]
        patch_area_h1 = h1.shape[2] * h1.shape[3]
        fv_hd1 = torch.nan_to_num(torch.sum(hd1, (2, 3)) / patch_area_hd1)
        fv_h1 = torch.nan_to_num(torch.sum(h1, (2, 3)) / patch_area_h1)
        i_fv = 0
        batch_size = x_hist.shape[0]
        coords_all = torch.zeros((n_cells_total, 2)).to(self.device)
        for i_batch in range(batch_size):
            c_mask_all = nuclei_mask[i_batch]
            cids = torch.unique(c_mask_all, sorted=True)[1:]
            H = c_mask_all.shape[0]
            W = c_mask_all.shape[1]
            y_idx = torch.arange(H, device=self.device).unsqueeze(1).repeat(1, W)
            x_idx = torch.arange(W, device=self.device).unsqueeze(0).repeat(H, 1)
            for cid in cids:
                c_mask = torch.where(c_mask_all == cid, 1, 0)
                c_area = torch.sum(c_mask)
                if c_area.item() == 0:
                    continue
                c_fv_hd1 = c_mask * hd1[i_batch]
                c_fv_hd1 = torch.nan_to_num(torch.sum(c_fv_hd1, (1, 2)) / c_area)
                c_fv_h1 = c_mask * h1[i_batch]
                c_fv_h1 = torch.nan_to_num(torch.sum(c_fv_h1, (1, 2)) / c_area)
                all_fv[i_fv, :] = torch.cat((c_fv_hd1, c_fv_h1, fv_hd1[i_batch], fv_h1[i_batch]))
                all_area[i_fv] = c_area
                ys = (c_mask * y_idx).float()
                xs = (c_mask * x_idx).float()
                y_mean = torch.sum(ys) / c_area
                x_mean = torch.sum(xs) / c_area
                coords_all[i_fv, 0] = x_mean / W
                coords_all[i_fv, 1] = y_mean / H
                i_fv += 1
        embeddings = torch.nan_to_num(self.embed_hist(all_fv))
        for i_batch in range(batch_size):
            n_cells_batch = int(n_cells[i_batch])
            if i_batch == 0:
                if batch_ct is not None:
                    batch_ct_pc = batch_ct[0, :n_cells_batch]
                if batch_expr is not None:
                    batch_expr_pc = batch_expr[0, :n_cells_batch, :]
                if patch_ids is not None:
                    patch_ids_pc = patch_ids[0, :n_cells_batch]
                embeddings_patches = torch.mean(embeddings[:n_cells_batch, :], 0)
                embeddings_patches = embeddings_patches.unsqueeze(0)
            else:
                if batch_ct is not None:
                    batch_ct_pc = torch.cat((batch_ct_pc, batch_ct[i_batch, :n_cells_batch]), 0)
                if batch_expr is not None:
                    batch_expr_pc = torch.cat((batch_expr_pc, batch_expr[i_batch, :n_cells_batch, :]), 0)
                if patch_ids is not None:
                    patch_ids_pc = torch.cat((patch_ids_pc, patch_ids[i_batch, :n_cells_batch]), 0)
                sum_idx_prev = torch.sum(n_cells[:i_batch]).item()
                embeddings_sample = torch.mean(embeddings[sum_idx_prev : sum_idx_prev + n_cells_batch, :], 0)
                embeddings_sample = embeddings_sample.unsqueeze(0)
                embeddings_patches = torch.cat((embeddings_patches, embeddings_sample), 0)
        if self.use_neighb:
            comp_estimated = self.estimate_comp(embeddings_patches)
            comp_tiled_all = None
            for i_batch in range(n_cells.shape[0]):
                n_cells_batch = int(n_cells[i_batch])
                if n_cells_batch > 0:
                    comp_tiled = torch.tile(comp_estimated[i_batch, :], (n_cells_batch, 1))
                    if comp_tiled_all is None:
                        comp_tiled_all = comp_tiled.clone()
                    else:
                        comp_tiled_all = torch.cat((comp_tiled_all, comp_tiled), 0)
        else:
            comp_estimated = None
            comp_tiled_all = None
        if self.use_celltype:
            out_cell_type, _ = self.mlp_hist(embeddings)
        else:
            out_cell_type = None
        if batch_expr is None:
            batch_expr_pc = None
        if batch_ct is None:
            if out_cell_type is not None:
                batch_ct_pc = torch.zeros(out_cell_type.shape).to(self.device)
            else:
                batch_ct_pc = None
        if patch_ids is None:
            patch_ids_pc = None
        if patch_keys is None:
            patch_keys_list = []
        else:
            patch_keys_list = list(patch_keys)
        if self.uni_provider.enable:
            u_patch = self.uni_provider.get_patch_features(patch_keys_list)
            u_cell_list = []
            idx_start = 0
            for i_batch in range(batch_size):
                n_cells_batch = int(n_cells[i_batch])
                if n_cells_batch > 0:
                    u_cell_list.append(u_patch[i_batch].unsqueeze(0).repeat(n_cells_batch, 1))
                idx_start += n_cells_batch
            if len(u_cell_list) > 0:
                u_cell = torch.cat(u_cell_list, dim=0)
            else:
                u_cell = torch.zeros((embeddings.shape[0], u_patch.shape[-1]), device=self.device)
            if self.fusion_type == "film":
                z, gamma, beta = self.fusion(embeddings, u_cell)
            else:
                z, g, up = self.fusion(embeddings, u_cell)
            fused_embeddings = getattr(self, "uni_blend_alpha", 1.0) * z + (1.0 - getattr(self, "uni_blend_alpha", 1.0)) * embeddings
            uni_expr_hint = self.uni_hint_head(torch.tanh(torch.nan_to_num(fused_embeddings)))
        else:
            fused_embeddings = embeddings
            uni_expr_hint = None
        self.last_embeddings = torch.nan_to_num(embeddings)
        self.last_coords = torch.nan_to_num(coords_all)
        expr_pred, traj = self.flow_forward(embeddings, coords_all, batch_expr_pc, n_cells, stage_pretrain=stage_pretrain)
        out_expr = torch.relu(expr_pred)
        out_expr_immune = out_expr
        out_expr_invasive = out_expr
        if self.use_celltype:
            out_cell_type_expr, fv_cell_type_expr = self.mlp_genes(out_expr)
        else:
            out_cell_type_expr = None
            fv_cell_type_expr = None
        if do_st_mlp is False:
            out_cell_type_gt_expr = None
            fv_cell_type_gt_expr = None
        elif batch_expr is not None and self.use_celltype:
            out_cell_type_gt_expr, fv_cell_type_gt_expr = self.mlp_genes(batch_expr_pc)
        else:
            out_cell_type_gt_expr = None
            fv_cell_type_gt_expr = None
        return (
            out_cell_type,
            out_map,
            batch_ct_pc,
            out_expr,
            out_expr_immune,
            out_expr_invasive,
            uni_expr_hint,
            out_cell_type_expr,
            fv_cell_type_expr,
            out_cell_type_gt_expr,
            fv_cell_type_gt_expr,
            batch_expr_pc,
            comp_estimated,
            all_area,
            patch_ids_pc,
        )

    def flow_forward(self, embeddings, coords_all, batch_expr_pc, n_cells, stage_pretrain=False):
        N = embeddings.shape[0]
        if os.environ.get("HISTCFM_TRAIN_DEBUG", "").strip() == "1":
            try:
                print(f"DEBUG: flow_forward training={self.training} has_cell_gt={batch_expr_pc is not None}")
            except Exception:
                pass
        if self.training and batch_expr_pc is not None:
            Y_clean = batch_expr_pc
            if stage_pretrain:
                t_vec = torch.zeros(N, 1, device=self.device)
                Y_t = Y_clean
            else:
                Y0 = self.flow_expr.sample_prior(N)
                t_vec = torch.rand(N, 1, device=self.device)
                noise = self.flow_noise_train_sigma * torch.randn_like(Y0)
                Y_t = t_vec * Y_clean + (1 - t_vec) * (Y0 + noise)
            knn_graph = build_knn_graph(torch.nan_to_num(coords_all), k=self.flow_k_neighbors)
            Y_hat = self.flow_expr.forward_step(embeddings, torch.nan_to_num(coords_all), torch.nan_to_num(Y_t), t_vec, knn_graph)
            return Y_hat, None
        else:
            Y_t = self.flow_expr.sample_prior(N)
            Y_t = Y_t + self.flow_noise_infer_sigma * torch.randn_like(Y_t)
            traj = [Y_t]
            knn_graph = build_knn_graph(torch.nan_to_num(coords_all), k=self.flow_k_neighbors)
            S = self.flow_steps
            for s in range(S):
                t1 = float(s) / float(S)
                t2 = float(s + 1) / float(S)
                Y_hat = self.flow_expr.forward_step(embeddings, coords_all, Y_t, torch.tensor(t1, device=self.device), knn_graph)
                if s == S - 1:
                    traj.append(Y_hat)
                    return Y_hat, traj
                Y_t = Y_t + (Y_hat - Y_t) / (1 - t1) * (t2 - t1)
                traj.append(Y_t)


# Internal compatibility alias for research scripts and later checkpoint tests.
# The supported public class is HistCFM.
Framework = HistCFM
