from torch import nn,log

class TSNE(nn.Module):
    def __init__(self, n_points, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super().__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_points, n_dim)

    def forward(self, pij, i, j):
        yi = self.logits(i)
        yj = self.logits(j)

        dist_sq = ((yi - yj) ** 2).sum(dim=-1)

        similarity = 1.0 / (1.0 + dist_sq)

        qij = similarity / similarity.sum()

        pij = pij / pij.sum()

        eps = 1e-12
        pij = pij.clamp_min(eps)
        qij = qij.clamp_min(eps)

        loss_kld = pij * log(pij / qij)

        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)
