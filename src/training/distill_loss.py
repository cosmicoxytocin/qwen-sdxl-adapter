"""Representation Inheritance Distillation Loss for Qwen-to-SDXL Adapter Training."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Distills knowledge from a frozen CLIP teacher to a student adapter using  Instance Semantic Alignment
    and Embedding Structure Alignment.
    """

    def __init__(self, struct_weight: float = 1.0):
        super().__init__()
        self.struct_weight = struct_weight
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        # 1. Instance Semantic Alignment (MSE loss)
        l_ins = F.mse_loss(student_features, teacher_features, reduction='mean')

        # Flatten seq dims for pairwise distance computation
        if student_features.dim() > 2:
            b = student_features.size(0)
            s_flat = student_features.view(b, -1)
            t_flat = teacher_features.view(b, -1)
        else:
            s_flat = student_features
            t_flat = teacher_features
        
        #2. Embedding Structure Alignment
        s_dist = torch.cdist(s_flat, s_flat, p=2.0)
        t_dist = torch.cdist(t_flat, t_flat, p=2.0)

        # Minimize the difference between the student's geometry and the teacher's geometry
        l_struct = F.mse_loss(s_dist, t_dist, reduction='mean')

        # Total loss is a weighted sum of instance and structure losses
        total_loss = l_ins + (self.struct_weight * l_struct)

        metrics = {
            "loss/total": total_loss.item(),
            "loss/l_ins": l_ins.item(),
            "loss/l_struct": l_struct.item(),
        }
        return total_loss, metrics
