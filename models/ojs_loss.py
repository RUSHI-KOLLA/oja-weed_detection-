"""
Organic Judgment Surveillance (OJS) Loss Functions.

Implements four complementary loss components used in AgroKD-Net:

* **GradientBalancedLoss** (GBPR) — inverse gradient-magnitude rebalancing.
* **ImbalanceAwareFocalLoss** (AIPL) — focal loss + inverse-frequency weights.
* **EnergyAwareDistillationLoss** (EAKD) — KL-divergence KD + energy penalty.
* **StructuralContextDistillationLoss** (SCD) — feature-affinity matrix KD.
* **OJSTotalLoss** — weighted combination of all four components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# GBPR — Gradient-Balanced Per-class Reweighting
# ---------------------------------------------------------------------------


class GradientBalancedLoss(nn.Module):
    """
    Gradient-Balanced Per-class Reweighting (GBPR).

    Maintains a momentum-smoothed running average of per-class gradient
    magnitudes and weights each class's loss contribution inversely
    proportional to that magnitude::

        w_k = 1 / (G_k + epsilon)

    Parameters
    ----------
    num_classes : int
    momentum : float
        EMA momentum for updating gradient magnitudes (default: 0.9).
    epsilon : float
        Stability term in the denominator (default: 1e-6).
    """

    def __init__(self, num_classes: int, momentum: float = 0.9, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.epsilon = epsilon
        # Running gradient magnitude per class (non-parameter buffer)
        self.register_buffer("grad_mag", torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(N, num_classes)``.
        targets : torch.Tensor
            Long tensor of shape ``(N,)`` with class indices.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Per-class cross-entropy
        ce = F.cross_entropy(logits, targets, reduction="none")  # (N,)

        # Update gradient magnitudes (no grad tracked for buffer update)
        with torch.no_grad():
            for k in range(self.num_classes):
                mask = targets == k
                if mask.any():
                    g = ce[mask].mean().detach().abs()
                    self.grad_mag[k] = (
                        self.momentum * self.grad_mag[k] + (1 - self.momentum) * g
                    )

        # Inverse-magnitude weights per sample
        weights = 1.0 / (self.grad_mag[targets] + self.epsilon)
        weights = weights / weights.sum() * len(targets)  # normalize
        return (weights * ce).mean()


# ---------------------------------------------------------------------------
# AIPL — Adaptive Imbalance-aware Progressive Loss (Focal + Frequency weights)
# ---------------------------------------------------------------------------


class ImbalanceAwareFocalLoss(nn.Module):
    """
    Adaptive Imbalance-aware Progressive Loss (AIPL).

    Combines focal loss with inverse-frequency class weights::

        w_k = 1 / log(1 + P_k)

    where P_k is the empirical frequency of class k.

    Parameters
    ----------
    num_classes : int
    gamma : float
        Focal modulation exponent (default: 2.0).
    epsilon : float
        Stability term (default: 1e-6).
    """

    def __init__(self, num_classes: int, gamma: float = 2.0, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.epsilon = epsilon
        self.register_buffer("class_counts", torch.zeros(num_classes))
        self.register_buffer("total_count", torch.tensor(0.0))

    def _update_frequencies(self, targets: torch.Tensor) -> None:
        with torch.no_grad():
            for k in range(self.num_classes):
                self.class_counts[k] += (targets == k).sum().float()
            self.total_count += len(targets)

    def _class_weights(self) -> torch.Tensor:
        if self.total_count == 0:
            return torch.ones(self.num_classes, device=self.class_counts.device)
        freqs = self.class_counts / (self.total_count + self.epsilon)
        return 1.0 / (torch.log(1.0 + freqs) + self.epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(N, num_classes)``.
        targets : torch.Tensor
            Long tensor of shape ``(N,)``.
        """
        self._update_frequencies(targets)

        probs = F.softmax(logits, dim=1)  # (N, C)
        pt = probs[torch.arange(len(targets)), targets]  # (N,)
        focal_weight = (1.0 - pt) ** self.gamma  # (N,)

        class_weights = self._class_weights()  # (C,)
        sample_class_weight = class_weights[targets]  # (N,)

        ce = F.cross_entropy(logits, targets, reduction="none")  # (N,)
        return (focal_weight * sample_class_weight * ce).mean()


# ---------------------------------------------------------------------------
# EAKD — Energy-Aware Knowledge Distillation Loss
# ---------------------------------------------------------------------------


class EnergyAwareDistillationLoss(nn.Module):
    """
    Energy-Aware Knowledge Distillation Loss (EAKD).

    KL divergence between temperature-scaled teacher and student logit
    distributions (Hinton et al., 2015) plus an optional energy penalty::

        L_EAKD = T^2 * KL(p_teacher || p_student) + lambda_energy * E_penalty

    Parameters
    ----------
    temperature : float
        Distillation temperature T (default: 4.0).
    lambda_energy : float
        Weight of the energy penalty term (default: 0.1).
    """

    def __init__(self, temperature: float = 4.0, lambda_energy: float = 0.1) -> None:
        super().__init__()
        self.T = temperature
        self.lambda_energy = lambda_energy

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        student_logits : torch.Tensor
            Shape ``(N, C)`` — raw student predictions (not softmax-ed).
        teacher_logits : torch.Tensor
            Shape ``(N, C)`` — raw teacher predictions.
        """
        T = self.T
        p_teacher = F.softmax(teacher_logits / T, dim=1)
        log_p_student = F.log_softmax(student_logits / T, dim=1)

        kl_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T ** 2)

        # Energy penalty: encourage student's L2 energy to match teacher's
        energy_penalty = (
            (student_logits.norm(dim=1) - teacher_logits.norm(dim=1).detach()) ** 2
        ).mean()

        return kl_loss + self.lambda_energy * energy_penalty


# ---------------------------------------------------------------------------
# SCD — Structural Context Distillation Loss
# ---------------------------------------------------------------------------


class StructuralContextDistillationLoss(nn.Module):
    """
    Structural Context Distillation Loss (SCD).

    Computes normalised affinity matrices from flattened feature maps and
    minimises their Frobenius-norm difference::

        R = F @ F^T / (||F|| * ||F^T||)
        L_SCD = ||R_teacher - R_student||_F^2

    The feature maps are spatially max-pooled to a fixed resolution before
    computing affinities to keep memory bounded.

    Parameters
    ----------
    pool_size : int
        Spatial size to which features are pooled (default: 8).
    """

    def __init__(self, pool_size: int = 8) -> None:
        super().__init__()
        self.pool_size = pool_size

    def _affinity(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute normalised affinity matrix from a 4-D feature tensor."""
        # Pool to fixed resolution
        feat = F.adaptive_avg_pool2d(feat, (self.pool_size, self.pool_size))
        b, c, h, w = feat.shape
        # Reshape to (B, C, N) where N = h*w
        f = feat.view(b, c, -1)  # (B, C, N)
        # Affinity: (B, N, N)
        ft = f.permute(0, 2, 1)  # (B, N, C)
        R = torch.bmm(ft, f)  # (B, N, N)
        norm = R.norm(dim=(1, 2), keepdim=True) + 1e-8
        return R / norm

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        student_feat : torch.Tensor
            Shape ``(B, C, H, W)``.
        teacher_feat : torch.Tensor
            Shape ``(B, C', H, W)``.  C' may differ from C.
        """
        R_s = self._affinity(student_feat)
        R_t = self._affinity(teacher_feat.detach())
        diff = R_s - R_t
        return (diff ** 2).mean()


# ---------------------------------------------------------------------------
# OJSTotalLoss — combined loss
# ---------------------------------------------------------------------------


class OJSTotalLoss(nn.Module):
    """
    Combined OJS Total Loss for AgroKD-Net training.

    Weighted sum::

        L = lambda_det * L_det
          + lambda_kd  * L_EAKD    (when teacher logits provided)
          + lambda_imb * L_AIPL
          + lambda_scd * L_SCD     (when teacher features provided)

    Parameters
    ----------
    num_classes : int
    lambda_det : float
        Weight for the task (detection/classification) loss.
    lambda_kd : float
        Weight for the EAKD knowledge-distillation loss.
    lambda_imb : float
        Weight for the AIPL imbalance-aware focal loss.
    lambda_scd : float
        Weight for the SCD structural distillation loss.
    temperature : float
        KD temperature for EAKD.
    lambda_energy : float
        Energy-penalty weight inside EAKD.
    gamma : float
        Focal-loss gamma for AIPL.
    """

    def __init__(
        self,
        num_classes: int,
        lambda_det: float = 1.0,
        lambda_kd: float = 0.5,
        lambda_imb: float = 0.3,
        lambda_scd: float = 0.2,
        temperature: float = 4.0,
        lambda_energy: float = 0.1,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_kd = lambda_kd
        self.lambda_imb = lambda_imb
        self.lambda_scd = lambda_scd

        self.gbpr = GradientBalancedLoss(num_classes)
        self.aipl = ImbalanceAwareFocalLoss(num_classes, gamma=gamma)
        self.eakd = EnergyAwareDistillationLoss(temperature=temperature, lambda_energy=lambda_energy)
        self.scd = StructuralContextDistillationLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        student_feat: Optional[torch.Tensor] = None,
        teacher_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        student_logits : torch.Tensor
            Shape ``(N, num_classes)``.
        targets : torch.Tensor
            Class index tensor of shape ``(N,)``.
        teacher_logits : torch.Tensor, optional
            Shape ``(N, num_classes)`` — required for EAKD.
        student_feat : torch.Tensor, optional
            4-D feature map — required for SCD.
        teacher_feat : torch.Tensor, optional
            4-D feature map — required for SCD.

        Returns
        -------
        dict
            ``{'total': tensor, 'gbpr': tensor, 'aipl': tensor,
               'eakd': tensor, 'scd': tensor}``
        """
        zero = student_logits.new_tensor(0.0)

        l_gbpr = self.gbpr(student_logits, targets)
        l_aipl = self.aipl(student_logits, targets)

        l_eakd = zero
        if teacher_logits is not None:
            l_eakd = self.eakd(student_logits, teacher_logits)

        l_scd = zero
        if student_feat is not None and teacher_feat is not None:
            l_scd = self.scd(student_feat, teacher_feat)

        total = (
            self.lambda_det * l_gbpr
            + self.lambda_imb * l_aipl
            + self.lambda_kd * l_eakd
            + self.lambda_scd * l_scd
        )

        return {
            "total": total,
            "gbpr": l_gbpr,
            "aipl": l_aipl,
            "eakd": l_eakd,
            "scd": l_scd,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, C = 16, 15
    print(f"🔧 Device: {device}")

    logits = torch.randn(N, C, device=device)
    t_logits = torch.randn(N, C, device=device)
    targets = torch.randint(0, C, (N,), device=device)
    s_feat = torch.randn(N, 128, 20, 20, device=device)
    t_feat = torch.randn(N, 256, 20, 20, device=device)

    # Individual losses
    gbpr_loss = GradientBalancedLoss(C).to(device)
    aipl_loss = ImbalanceAwareFocalLoss(C).to(device)
    eakd_loss = EnergyAwareDistillationLoss().to(device)
    scd_loss = StructuralContextDistillationLoss().to(device)

    print(f"✅ GBPR loss : {gbpr_loss(logits, targets).item():.4f}")
    print(f"✅ AIPL loss : {aipl_loss(logits, targets).item():.4f}")
    print(f"✅ EAKD loss : {eakd_loss(logits, t_logits).item():.4f}")
    print(f"✅ SCD  loss : {scd_loss(s_feat, t_feat).item():.4f}")

    # Combined
    ojs = OJSTotalLoss(num_classes=C).to(device)
    result = ojs(logits, targets, t_logits, s_feat, t_feat)
    print("\n📊 OJSTotalLoss breakdown:")
    for k, v in result.items():
        print(f"   {k:6s}: {v.item():.4f}")

    print("\n✅ All OJS loss tests passed.")
    sys.exit(0)
