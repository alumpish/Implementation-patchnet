import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)


class AdMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m_l=0.4, m_s=0.1):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = [m_s, m_l]
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels, is_spoof):
        '''
        input: 
            x shape (N, in_features)
            labels shape (N)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        m = torch.tensor([self.m[ele] for ele in is_spoof]).to(x.device)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels.argmax(dim=1)]) - m)
        excl = torch.cat([torch.cat((wf[i, :y.argmax()], wf[i, y.argmax()+1:])).unsqueeze(0)
                         for i, y in enumerate(is_spoof)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)

        L = numerator - torch.log(denominator)

        return - torch.mean(L)


class PatchLoss(nn.Module):
    def __init__(self, num_classes, alpha1=1.0, alpha2=1.0):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sim_loss = SimilarityLoss()
        self.amsm_loss = AdMSoftmaxLoss(512, num_classes)

    def forward(self, x1, x2, label, is_spoof):
        amsm_loss1 = self.amsm_loss(x1.squeeze(), label.type(torch.long).squeeze(), is_spoof)
        amsm_loss2 = self.amsm_loss(x2.squeeze(), label.type(torch.long).squeeze(), is_spoof)
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        sim_loss = self.sim_loss(x1, x2)
        loss = self.alpha1 * sim_loss + self.alpha2 * (amsm_loss1 + amsm_loss2)

        return loss
