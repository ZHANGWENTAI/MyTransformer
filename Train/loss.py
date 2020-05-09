import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.vocab_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 而不是 -1 是因为 <pad> 的概率为 0
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.tensor(true_dist, requires_grad=False))

class LossComputation:

    def __init__(self, projector, criterion, optimizer=None):
        self.projector = projector
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, label, norm):
        x = self.projector(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              label.contiguous().view(-1)) / norm
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optim.zero_grad()
        return loss.item() * norm.float()