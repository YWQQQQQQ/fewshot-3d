from typing import Tuple

import torch
from torch import nn, Tensor


def get_sp_sn(sim_mat, full_edge, sp_edge_mask, qry_edge_mask):
    sp = sim_mat*full_edge[:,0]
    sn = sim_mat*full_edge[:,1]
    sp_sp = (sp * sp_edge_mask).view(-1)
    sp_sp = sp_sp[sp_sp>0]
    sp_sn = (sn * sp_edge_mask).view(-1)
    sp_sn = sp_sn[sp_sn>0]
    qry_sp = (sp * qry_edge_mask).view(-1)
    qry_sp = qry_sp[qry_sp>0]
    qry_sn = (sn * qry_edge_mask).view(-1)
    qry_sn = qry_sn[qry_sn>0]

    return sp_sp, sp_sn,qry_sp,qry_sn


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)) / 2

        return loss

if __name__ == "__main__":
    logits_layer = nn.functional.normalize(torch.rand(25,2,26,26, requires_grad=True))
    full_edge = torch.randint(low=0,high=2, size=(25,2,26,26))
    sp_edge_mask = torch.zeros(25,26,26)
    sp_edge_mask[:, :-1, :-1] = 1
    qry_edge_mask = 1 - sp_edge_mask
    sp_sp, sp_sn, qry_sp, qry_sn = get_sp_sn(logits_layer[:,0], full_edge, sp_edge_mask, qry_edge_mask)

    criterion = CircleLoss(m=0.25, gamma=256)
    circle_loss = criterion(sp_sp, sp_sn)

    print(circle_loss)
    circle_loss = criterion(qry_sp, qry_sn)
    print(circle_loss)
