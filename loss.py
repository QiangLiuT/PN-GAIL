import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class CULoss(nn.Module):
    def __init__(self, beta, non=False):
        super(CULoss, self).__init__()
        self.loss = nn.SoftMarginLoss()
        self.beta = beta
        self.non = non

    def forward(self, conf, labeled, unlabeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))

        unlabeled_pos_risk = torch.mean(self.beta * self.loss(unlabeled, torch.ones(unlabeled.shape).to(device)))
        unlabeled_neg_risk = torch.mean(self.beta * self.loss(unlabeled, -torch.ones(unlabeled.shape).to(device)))
        pos_risk = torch.mean(conf * y_conf_pos)

        neg_risk = torch.mean((1 - conf) * y_conf_neg)
        other_pos_risk = torch.mean((- self.beta) * y_conf_pos) + unlabeled_pos_risk
        other_neg_risk = torch.mean((- self.beta) * y_conf_neg) + unlabeled_neg_risk

        if self.non:
            other_risk = other_pos_risk + other_neg_risk
            other_risk = torch.clamp(other_risk, min=-0.01, max=0.01)
            objective = neg_risk + pos_risk + other_risk

        else:
            objective = neg_risk + pos_risk

        return objective


class PNLoss(nn.Module):
    def __init__(self):
        super(PNLoss, self).__init__()
        self.loss = nn.SoftMarginLoss()

    def forward(self, conf, labeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))

        objective = torch.mean(conf * y_conf_pos + (1 - conf) * y_conf_neg)
        return objective
