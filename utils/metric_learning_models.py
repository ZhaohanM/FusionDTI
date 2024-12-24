import logging
import os
import sys

sys.path.append("../")

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn import Module
from tqdm import tqdm
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

class FusionDTI(nn.Module):
    def __init__(self, prot_out_dim, drug_out_dim, args):
        super(FusionDTI, self).__init__()
        self.fusion = args.fusion
        self.drug_reg = nn.Linear(drug_out_dim, 512)
        self.prot_reg = nn.Linear(prot_out_dim, 512)

        if self.fusion == "CAN":
            self.can_layer = CAN_Layer(hidden_dim=512, num_heads=8, args=args)
            self.mlp_classifier = MlPdecoder_CAN(input_dim=1024)
        elif self.fusion == "BAN":
            self.ban_layer = weight_norm(BANLayer(512, 512, 256, 2), name='h_mat', dim=None)
            self.mlp_classifier = MlPdecoder_CAN(input_dim=256)
        elif self.fusion == "Nan":
            self.mlp_classifier_nan = MlPdecoder_CAN(input_dim=1246)

    def forward(self, prot_embed, drug_embed, prot_mask, drug_mask):
        if self.fusion == "Nan":
            prot_embed = prot_embed.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embed.mean(1)  # query : [batch_size, hidden]
            joint_embed = torch.cat([prot_embed, drug_embed], dim=1)
            score = self.mlp_classifier_nan(joint_embed)
        else:
            prot_embed = self.prot_reg(prot_embed)
            drug_embed = self.drug_reg(drug_embed)

            if self.fusion == "CAN":
                joint_embed = self.can_layer(prot_embed, drug_embed, prot_mask, drug_mask)
            elif self.fusion == "BAN":
                joint_embed, att = self.ban_layer(prot_embed, drug_embed)

            score = self.mlp_classifier(joint_embed)

        return score

# class FusionDTI(nn.Module):
#     def __init__(self, prot_out_dim, disease_out_dim, args):
#         super(FusionDTI, self).__init__()
#         """Constructor for the model.

#         Args:
#             prot_out_dim (_type_): Dimension of the protein encoder.
#             disease_out_dim (_type_): Dimension of the drug encoder.
#             args (_type_): _description_
#         """
        
#         self.fusion = args.fusion
#         self.drug_reg = nn.Linear(disease_out_dim, 512)
#         self.prot_reg = nn.Linear(prot_out_dim, 512)
        
#         if self.fusion == "CAN":
#             self.can_layer = CAN_Layer(hidden_dim=512, num_heads=8, args=args)
#             self.mlp_classifier = MlPdecoder_CAN(input_dim=1024)
#         elif self.fusion == "BAN":
#             self.ban_layer = weight_norm(BANLayer(512, 512, 256, 2), name='h_mat', dim=None)
#             self.mlp_classifier = MlPdecoder_CAN(input_dim=256)
#             # self.mlp_classifier = MLPdecoder_BAN(in_dim=256, hidden_dim=512, out_dim=128)

#     def forward(self, prot_embed, drug_embed, prot_mask, drug_mask):
#         prot_embed = self.prot_reg(prot_embed)
#         drug_embed = self.drug_reg(drug_embed)
        
#         if self.fusion == "CAN":
#             joint_embed = self.can_layer(prot_embed, drug_embed, prot_mask, drug_mask)
#         elif self.fusion == "BAN":
#             joint_embed, att = self.ban_layer(prot_embed, drug_embed)
        
#         score = self.mlp_classifier(joint_embed)
#         return score
    
#     def count_parameters(self):
#         """Counts the number of trainable parameters for the active fusion module."""
#         if self.fusion == "CAN":
#             return sum(p.numel() for p in self.can_layer.parameters() if p.requires_grad)
#         elif self.fusion == "BAN":
#             return sum(p.numel() for p in self.ban_layer.parameters() if p.requires_grad)
#         else:
#             raise ValueError("Unsupported fusion type")
    
class Pre_encoded(nn.Module):
    def __init__(
            self, prot_encoder, drug_encoder, args
    ):
        """Constructor for the model.

        Args:
            prot_encoder (_type_): Protein sturcture-aware sequence encoder.
            drug_encoder (_type_): Drug SFLFIES encoder.
            args (_type_): _description_
        """
        super(Pre_encoded, self).__init__()
        self.prot_encoder = prot_encoder
        self.drug_encoder = drug_encoder
        
    def encoding(self, prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask):
        # Process protein encoder with hidden state output
        prot_embed = self.prot_encoder(
            input_ids=prot_input_ids, 
            attention_mask=prot_attention_mask, 
            output_hidden_states=True,  # Request hidden states
            return_dict=True
        ).hidden_states[-1]

        drug_embed = self.drug_encoder(
            input_ids=drug_input_ids, attention_mask=drug_attention_mask, return_dict=True
        ).last_hidden_state

        return prot_embed, drug_embed
    
    
class CAN_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, args):
        super(CAN_Layer, self).__init__()
        self.agg_mode = args.agg_mode
        self.group_size = args.group_size  #  Control Fusion Scale
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def group_embeddings(self, x, mask, group_size):
        N, L, D = x.shape
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        mask_grouped = mask.view(N, groups, group_size).any(dim=2)
        return x_grouped, mask_grouped

    def forward(self, protein, drug, mask_prot, mask_drug):
        # Group embeddings before applying multi-head attention
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size)
        drug_grouped, mask_drug_grouped = self.group_embeddings(drug, mask_drug, self.group_size)
        
        # print("protein_grouped:", protein_grouped.shape)
        # print("mask_prot_grouped:", mask_prot_grouped.shape)

        # Compute queries, keys, values for both protein and drug after grouping
        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads, self.head_size)
        key_prot = self.apply_heads(self.key_p(protein_grouped), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads, self.head_size)

        query_drug = self.apply_heads(self.query_d(drug_grouped), self.num_heads, self.head_size)
        key_drug = self.apply_heads(self.key_d(drug_grouped), self.num_heads, self.head_size)
        value_drug = self.apply_heads(self.value_d(drug_grouped), self.num_heads, self.head_size)

        # Compute attention scores
        logits_pp = torch.einsum('blhd, bkhd->blkh', query_prot, key_prot)
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_drug)
        logits_dp = torch.einsum('blhd, bkhd->blkh', query_drug, key_prot)
        logits_dd = torch.einsum('blhd, bkhd->blkh', query_drug, key_drug)
        # print("logits_pp:", logits_pp.shape)

        alpha_pp = self.alpha_logits(logits_pp, mask_prot_grouped, mask_prot_grouped)
        alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_drug_grouped)
        alpha_dp = self.alpha_logits(logits_dp, mask_drug_grouped, mask_prot_grouped)
        alpha_dd = self.alpha_logits(logits_dd, mask_drug_grouped, mask_drug_grouped)

        prot_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)) / 2
        drug_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha_dd, value_drug).flatten(-2)) / 2
        
        # print("prot_embedding:", prot_embedding.shape)
        
        # Continue as usual with the aggregation mode
        if self.agg_mode == "cls":
            prot_embed = prot_embedding[:, 0]  # query : [batch_size, hidden]
            drug_embed = drug_embedding[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            prot_embed = prot_embedding.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embedding.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            prot_embed = (prot_embedding * mask_prot_grouped.unsqueeze(-1)).sum(1) / mask_prot_grouped.sum(-1).unsqueeze(-1)
            drug_embed = (drug_embedding * mask_drug_grouped.unsqueeze(-1)).sum(1) / mask_drug_grouped.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
            
        # print("prot_embed:", prot_embed.shape)

        query_embed = torch.cat([prot_embed, drug_embed], dim=1)

        # print("query_embed:", query_embed.shape)
        return query_embed   

class MlPdecoder_CAN(nn.Module):
    def __init__(self, input_dim):
        super(MlPdecoder_CAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, input_dim // 4)
        self.bn3 = nn.BatchNorm1d(input_dim // 4)
        self.output = nn.Linear(input_dim // 4, 1)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.bn2(torch.relu(self.fc2(x)))
        x = self.bn3(torch.relu(self.fc3(x)))
        x = torch.sigmoid(self.output(x))
        return x
    
class MLPdecoder_BAN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPdecoder_BAN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        # x = self.fc4(x)
        x = torch.sigmoid(self.fc4(x)) 
        return x

class BANLayer(nn.Module):
    """ Bilinear attention network
    Modified from https://github.com/peizhenbai/DrugBAN/blob/main/ban.py
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        # print("v_num", v_num)
        # print("v_num ", v_num)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            # print("v_", v_.shape)
            # print("q_ ", q_.shape)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            # print("Attention map_1",att_maps.shape)
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
            # print("Attention map_2",att_maps.shape)
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
            # print("Attention map_softmax", att_maps.shape)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

    
class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
class BatchFileDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        batch_file = self.file_list[idx]
        data = torch.load(batch_file)
        return data['prot'], data['drug'], data['prot_mask'], data['drug_mask'], data['y']
