import torch


class log_variance_loss(torch.nn.Module):
    def __init__(self, N):
        super(log_variance_loss, self).__init__()
        self.N = N

    def forward(self, uncertainty):
        d = self.N * uncertainty + self.N
        d = d.mean()
        return d


class L1_uncertainty_loss(torch.nn.Module):
    def __init__(self, N):
        super(L1_uncertainty_loss, self).__init__()
        self.N = N

    def forward(self, input, uncertainty, target):
        #d = torch.exp(-uncertainty) * torch.abs(input - target) + (uncertainty+1)/2.0

        # https://arxiv.org/pdf/1703.04977.pdf
        # Equation 8 - uncertainty is log(sigma^2)

        # uncertainty lies between -1 and +1 (by tanh)
        # scale to [-N, N]
        # add N so loss term is positive (simpler for math)
        # units are in log-variance, so [-N, N] corresponds to [exp(-N), exp(+N)]

        #N = 7.0
        #d = torch.exp(-N * uncertainty) * torch.abs(input - target) + N * uncertainty + N
        d = torch.exp(-self.N * uncertainty) * torch.abs(input - target)

        # real valued
        # d = torch.exp(-uncertainty) * torch.abs(input - target) + uncertainty

        # weight different parts?
        # d = torch.exp(-uncertainty) * torch.abs(input - target) + uncertainty * 0.01

        d = d.mean()

        return d

        # return torch.div(torch.norm(input - target, p=1), uncertainty ** 2) + torch.log(uncertainty ** 2)
